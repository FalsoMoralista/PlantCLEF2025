import sys
sys.path.append('../')
import torch
import os
os.environ['TORCH_DEBUG'] = 'DETAIL'
import timm

from src.helper import load_class_mapping, load_species_mapping
from datasets.pc2025 import build_test_dataset, build_test_transform
from visualizer import AttentionMapVisualizer

import faiss
import faiss.contrib.torch_utils

from src.models.custom_vision_transformer import PilotVisionTransformer
from PIL import Image
import random
import torch.nn.functional as F

import matplotlib.pyplot as plt

from torchvision import transforms

import csv

def visualize_global_attention_on_image(image, attn_map, patch_size=64, save_path=None):
    import torchvision.transforms.functional as TF
    from torchvision.utils import make_grid
    import numpy as np
    from PIL import Image


    """
    Visualizes global attention (mean over all tokens and heads) as a heatmap over the original image.

    image: PIL image (2048x2048)
    attn_map: Tensor of shape (1, num_heads, 1024, 1024)
    """
    assert attn_map.shape[-1] == attn_map.shape[-2], "Expected square attention map"

    # 1. Average over heads
    attn_map = attn_map.mean(dim=1)[0]  # Shape: [1024, 1024]

    # 2. Aggregate attention received by each patch (i.e., sum over rows → what each token receives)
    global_attention = attn_map.sum(dim=0)  # [1024]
    
    # 3. Normalize
    global_attention = (global_attention - global_attention.min()) / (global_attention.max() - global_attention.min())
    global_attention = global_attention.reshape(32, 32).detach().cpu().numpy()

    # 4. Resize to match image
    attn_map_resized = TF.resize(TF.to_pil_image(global_attention), image.size, interpolation=Image.BICUBIC)

    # 5. Overlay
    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    plt.imshow(attn_map_resized, cmap='jet', alpha=0.5)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def mask_image_from_attention(
    image, attn_map, patch_size=64, threshold=0.2, replace_mode='grayscale', save_path=None
):
    """
    Visualizes global attention as a heatmap and masks low-attention regions in the original image.

    image: PIL image (2048x2048)
    attn_map: Tensor of shape (1, num_heads, 1024, 1024)
    threshold: Float in [0, 1]. Attention scores below this are masked.
    replace_mode: 'grayscale' or 'black'
    """
    import torchvision.transforms.functional as TF
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image

    # 1. Compute mean attention over heads
    attn_map = attn_map.mean(dim=1)[0]  # Shape: [1024, 1024]

    # 2. Global attention received per patch (sum over rows)
    global_attention = attn_map.sum(dim=0)  # [1024]

    # 3. Normalize
    global_attention = (global_attention - global_attention.min()) / (global_attention.max() - global_attention.min())
    global_attention = global_attention.reshape(32, 32)  # Shape: (H, W)

    # 4. Upsample to full image size (2048x2048)
    global_attention_np = global_attention.unsqueeze(0).unsqueeze(0)  # [1, 1, 32, 32]
    attn_resized = torch.nn.functional.interpolate(global_attention_np, size=image.size[::-1], mode='bilinear')[0, 0]

    # 5. Convert to NumPy mask
    mask = (attn_resized >= threshold).cpu().numpy().astype(np.uint8)  # 1 = keep, 0 = mask out

    # 6. Apply mask to image
    img_np = np.array(image)

    if replace_mode == 'grayscale':
        gray = np.dot(img_np[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        gray = np.stack([gray]*3, axis=-1)
        masked_img = img_np * mask[..., None] + gray * (1 - mask[..., None])
    elif replace_mode == 'black':
        masked_img = img_np * mask[..., None]
    else:
        raise ValueError("Unsupported replace_mode: choose 'grayscale' or 'black'")

    # 7. Visualization
    plt.figure(figsize=(12, 12))
    plt.imshow(masked_img)
    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def get_high_attention_crop_positions(attn_scores, crop_size, image_size, threshold=0.2):
    """
    Returns the (x, y) pixel positions of patches with attention >= threshold.

    Args:
        attn_scores (Tensor): 2D tensor of shape (H_patches, W_patches)
        crop_size (int): Size of each patch in pixels
        image_size (tuple): (H, W) of original image in pixels
        threshold (float): Minimum attention score to keep a patch

    Returns:
        Tensor of shape (N, 2) with top-left corner pixel positions (x, y)
    """
    device = attn_scores.device
    num_patches_y, num_patches_x = attn_scores.shape
    H, W = image_size

    assert H // crop_size == num_patches_y and W // crop_size == num_patches_x, \
        "Image size and attention map shape are inconsistent with crop size."

    # Keep only high-attention scores
    high_attn_mask = (attn_scores.flatten() >= threshold)

    # Grid of patch top-left corners
    x_indices = torch.arange(0, num_patches_x, device=device) * crop_size
    y_indices = torch.arange(0, num_patches_y, device=device) * crop_size
    y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')
    positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)  # shape (N, 2)

    return positions[high_attn_mask]


def crop_at_positions(img, positions, crop_size, output_size):
    """
    Crops and resizes only the specified (x, y) positions from the image.

    Args:
        img (Tensor): Image tensor (C, H, W)
        positions (Tensor): Tensor of shape (N, 2) with top-left crop positions (x, y)
        crop_size (int): Size of crop
        output_size (int): Resize output size (e.g., 518)

    Returns:
        Tensor of cropped and resized patches (N, C, output_size, output_size)
    """
    C, H, W = img.shape       
    crops = []
    for (x, y) in positions:
        x, y = int(x), int(y)
        crop = img[:, y:y+crop_size, x:x+crop_size]
        if crop.shape[1] == crop_size and crop.shape[2] == crop_size:
            crops.append(crop)
    
    if not crops:
        return None  # no valid crops

    crops = torch.stack(crops)
    resized_crops = F.interpolate(
        crops,
        size=(output_size, output_size),
        mode='bilinear',
        align_corners=False,
        antialias=True
    )
    return resized_crops

def plot_crops_in_grid_positions(crops, positions, image_size, crop_size, save_path):
    """
    Reconstructs and plots an image grid where only the selected crops are placed
    at their original positions, and the rest is blank.

    Args:
        crops (Tensor): Selected crops of shape (N, C, crop_size, crop_size)
        positions (Tensor): Corresponding top-left (x, y) positions, shape (N, 2)
        image_size (tuple): Original image size (H, W)
        crop_size (int): Size of each crop in pixels
    """
    device = crops.device
    C = crops.shape[1]
    H, W = image_size
    blank_canvas = torch.zeros((C, H, W), dtype=crops.dtype, device=device)

    for crop, (x, y) in zip(crops, positions):
        x, y = int(x.item()), int(y.item())
        blank_canvas[:, y:y+crop_size, x:x+crop_size] = crop

    # Convert to numpy and transpose for display
    img_np = blank_canvas.permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # normalize for display

    plt.figure(figsize=(10, 10))
    plt.imshow(img_np)
    plt.axis('off')
    plt.title("Selected Crops in Original Grid Positions")
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()


pretrained_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar'
cid_to_spid = load_class_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/class_mapping.txt')
spid_to_sp = load_species_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/species_id_to_name.txt')
test_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/test_dataset/'
patch_size = 64
batch_size = 1
if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

images = os.listdir(test_dir+'test/')
#random.shuffle(images)
#images = images[0:20]
print('len images:', len(images))


#images = ['GUARDEN-AMB-PR13-1-2-20240417.jpg']
#images = ['CBN-can-A6-20230705.jpg']
epochs = [15] #, 20, 25, 40, 60, 80, 85]

model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
model.to(device)

dino_cls = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
dino_cls.to(device)

data_config = timm.data.resolve_model_data_config(model)
#transform = build_test_transform(data_config, n=64)

test_dataset, test_dataloader, dist_sampler = build_test_dataset(image_folder=test_dir,
                                            data_config=data_config,
                                            input_resolution=(2048,2048),
                                            num_workers=8,
                                            n=64,
                                            world_size=1,
                                            rank=0,
                                            shuffle=False,
                                            batch_size=1)

r_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/logs/experiment_0/'

std_transform = transforms.Compose([
    transforms.Resize((2048, 2048), interpolation=Image.BICUBIC),  # Optional: scale to be divisible by 16
    transforms.ToTensor(),  # Convert to tensor: (C, H, W)
    transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
])


predicted_items = []

for epoch_no in epochs:
    checkpoint = torch.load(r_path+f"experiment_0-ep{epoch_no}.pth.tar", map_location=torch.device('cpu'))
    ViT = PilotVisionTransformer(img_size=[2048,2048], pretrained_patch_embedder=model,patch_size=patch_size, embed_dim=768, depth=6)
    print('Loading Vision Transformer:', ViT)        

    pretrained_dict = checkpoint['custom_vit']
    msg = ViT.load_state_dict(pretrained_dict)
    print('Loading state_dict:', msg)
    ViT.linear = torch.nn.Identity()
    ViT.to(device)
    
    for itr, (img_tensor, _, image_name) in enumerate(test_dataloader):
        if itr % 100 == 0:
            print(itr,'/',len(test_dataset))
    #for i, image_name in enumerate(images):
    #image_name = images[0]
    #img = Image.open(f'../pretrained_models/{image_name}')
        img_tensor = img_tensor.to(device)
        image_name = image_name[0]
        img = Image.open(f'{test_dir}/test/{image_name}')
        img = std_transform(img).to(device)
        #img_tensor = transform(img)#.unsqueeze(0)
        #img_tensor = img_tensor.to(device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):        
            with torch.inference_mode():
                _, attn = ViT(img_tensor.squeeze(0), return_attention=True)
        #del img_tensor
        #print('X size:', x.size())
        #print('len attn:', len(attn))
        #print('attn size:', attn[0][1].size())
        #attn_map = (attn[5][1] + attn[4][1]+ attn[3][1] + attn[2][1] + attn[1][1] + attn[0][1])/6
        num_blocks = 6
        attn_map = torch.stack([attn[i][1] for i in range(num_blocks)]).mean(0)
        
        global_attention = attn_map.clone().detach()
        # 1. Average over heads
        global_attention = attn_map.mean(dim=1)[0]  # Shape: [1024, 1024]

        # 2. Aggregate attention received by each patch (i.e., sum over rows → what each token receives)
        global_attention = global_attention.sum(dim=0)  # [1024]

        # 3. Normalize
        eps = 1e-6
        global_attention = (global_attention - global_attention.min()) / (global_attention.max() - global_attention.min() + eps)
        global_attention = global_attention.reshape(32, 32) #.detach().cpu().numpy()

        high_attn_positions = get_high_attention_crop_positions(
            attn_scores=global_attention,
            crop_size=64,
            image_size=(2048, 2048),
            threshold=0.6
        )

        # Crop and resize selected regions
        selected_crops = crop_at_positions(
            img=img,                # tensor (C, 2048, 2048)
            positions=high_attn_positions, # pixel coordinates (x, y)
            crop_size=64,
            output_size=518
        )#.to(device)

        #print('selected_crops', selected_crops.size())
        B,C,H,W = selected_crops.size() 
        
        image_predictions = []
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            with torch.inference_mode():
                out = dino_cls(selected_crops)
        #del selected_crops
        #torch.cuda.empty_cache()

        #print('size: ', out.size())
        for output in out:
            top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=0) * 100, k=5)
            top5_probabilities = top5_probabilities.to(torch.float32).cpu().detach().numpy()
            top5_class_indices = top5_class_indices.to(torch.float32).cpu().detach().numpy()
            proba, cid = top5_probabilities[0], top5_class_indices[0]
            species_id = cid_to_spid[cid]
            species = spid_to_sp[species_id]
            if not species_id in image_predictions: #and proba >= 0.5:
                image_predictions.append(species_id)
        
        #print('Pred: ', image_name.replace('.jpg', ''), len(image_predictions))
        predicted_items.append((image_name.replace('.jpg', ''), image_predictions))
        #print(predicted_items)
    print('predicted items length:', len(predicted_items))

    with open('predictions_1.csv', mode='w', newline='') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['quadrat_id', 'species_ids'])  # Header

        for quadrat_id, species_ids in predicted_items:
            writer.writerow([quadrat_id, str(species_ids)])

        #visualize_global_attention_on_image(img, attn_map=attn_map, save_path=f'attention/pilot/empirical/{i}_pilot_epoch_{epoch_no}_all_blocks_AVG.jpg')
        #mask_image_from_attention(img,attn_map=attn_map,patch_size=64,threshold=0.55,replace_mode='black',save_path=f'attention/pilot/empirical/{i}_epoch_{epoch_no}_blocks_6_5_4_score=0.55.jpg')
        #plot_crops_in_grid_positions(
        #    crops=selected_crops,                 # shape (N, C, crop_size, crop_size)
        #    positions=high_attn_positions,        # shape (N, 2)
        #    image_size=(2048, 2048),
        #    crop_size=64,
        #    save_path=f'attention/pilot/empirical/{i}_epoch_{epoch_no}_crops_score=0.55.jpg'
        #)

        #mask_image_from_attention(img,attn_map=attn_map,patch_size=64,threshold=0.3,replace_mode='black',save_path=f'attention/pilot/empirical/{i}_epoch_{epoch_no}_blocks_6_5_4_score=0.3.jpg')
        #mask_image_from_attention(img,attn_map=attn_map,patch_size=64,threshold=0.4,replace_mode='black',save_path=f'attention/pilot/empirical/{i}epoch_{epoch_no}_blocks_6_5_4_score=0.4.jpg')
        #mask_image_from_attention(img,attn_map=attn_map,patch_size=64,threshold=0.5,replace_mode='black',save_path=f'attention/pilot/empirical/{i}epoch_{epoch_no}_blocks_6_5_4_score=0.5.jpg')
        #mask_image_from_attention(img,attn_map=attn_map,patch_size=64,threshold=0.6,replace_mode='black',save_path=f'attention/pilot/empirical/{i}epoch_{epoch_no}_blocks_6_5_4_score=0.6.jpg')
        
exit(0)


images = os.listdir(test_dir+'test/')


    #img = None
    #if 'https://' in args.image or 'http://' in  args.image:
    #    img = Image.open(urlopen(args.image))
    #elif args.image != None:
    #    img = Image.open(args.image)
        
    #if img != None:
    #    img = transforms(img).unsqueeze(0)
    #    img = img.to(device)
    #    output = model(img)  # unsqueeze single image into batch of 1
    #    top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
    #    top5_probabilities = top5_probabilities.cpu().detach().numpy()
    #    top5_class_indices = top5_class_indices.cpu().detach().numpy()

    #    for proba, cid in zip(top5_probabilities[0], top5_class_indices[0]):
    #        species_id = cid_to_spid[cid]
    #        species = spid_to_sp[species_id]
    #        print(species_id, species, proba)