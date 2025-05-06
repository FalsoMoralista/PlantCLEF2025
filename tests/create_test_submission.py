import sys
sys.path.append('../')
import torch
import os
os.environ['TORCH_DEBUG'] = 'DETAIL'
import timm

from src.helper import load_class_mapping
from datasets.pc2025 import build_test_dataset, build_test_transform
from visualizer import AttentionMapVisualizer

import faiss
import faiss.contrib.torch_utils

from src.models.custom_vision_transformer import PilotVisionTransformer
from PIL import Image

def visualize_global_attention_on_image(
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



pretrained_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar'
cid_to_spid = load_class_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/class_mapping.txt')
test_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/test_dataset/'
patch_size = 64
batch_size = 1
if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

images = ['GUARDEN-AMB-PR13-1-2-20240417.jpg']
images = ['CBN-can-A6-20230705.jpg']
epochs = [15]

model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction
model.to(device)

data_config = timm.data.resolve_model_data_config(model)
transform = build_test_transform(data_config, n=64)

r_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/logs/experiment_0/'

pilot = True

for epoch_no in epochs:
    checkpoint = torch.load(r_path+f"experiment_0-ep{epoch_no}.pth.tar", map_location=torch.device('cpu'))
    ViT = PilotVisionTransformer(img_size=[2048,2048], pretrained_patch_embedder=model,patch_size=patch_size, embed_dim=768, depth=6)
    print('Loading Vision Transformer:', ViT)        

    pretrained_dict = checkpoint['custom_vit']
    msg = ViT.load_state_dict(pretrained_dict)
    print('Loading state_dict:', msg)
    ViT.linear = torch.nn.Identity()
    ViT.to(device)

    image_name = images[0]
    img = Image.open(f'../pretrained_models/{image_name}')
    img_tensor = transform(img)#.unsqueeze(0)
    img_tensor = img_tensor.to(device)
    print('img tensor:', img_tensor.size())

    x, attn = ViT(img_tensor, return_attention=True)
    print('X size:', x.size())
    print('len attn:', len(attn))
    print('attn size:', attn[0][1].size())
    attn_map = (attn[5][1] + attn[4][1]+ attn[3][1])/3
    #visualize_global_attention_on_image(img, attn_map=attn_map,save_path=f'attention/pilot/pilot_epoch_{epoch_no}_block_6_5_AVG.jpg')
    visualize_global_attention_on_image(img,attn_map=attn_map,patch_size=64,threshold=0.3,replace_mode='black',save_path=f'attention/pilot/epoch_{epoch_no}_blocks_6_5_4_score=0.3.jpg')
    visualize_global_attention_on_image(img,attn_map=attn_map,patch_size=64,threshold=0.4,replace_mode='black',save_path=f'attention/pilot/epoch_{epoch_no}_blocks_6_5_4_score=0.4.jpg')
    visualize_global_attention_on_image(img,attn_map=attn_map,patch_size=64,threshold=0.5,replace_mode='black',save_path=f'attention/pilot/epoch_{epoch_no}_blocks_6_5_4_score=0.5.jpg')
    visualize_global_attention_on_image(img,attn_map=attn_map,patch_size=64,threshold=0.6,replace_mode='black',save_path=f'attention/pilot/epoch_{epoch_no}_blocks_6_5_4_score=0.6.jpg')
    visualize_global_attention_on_image(img,attn_map=attn_map,patch_size=64,threshold=0.65,replace_mode='black',save_path=f'attention/pilot/epoch_{epoch_no}_blocks_6_5_4_score=0.65.jpg')
    visualize_global_attention_on_image(img,attn_map=attn_map,patch_size=64,threshold=0.75,replace_mode='black',save_path=f'attention/pilot/epoch_{epoch_no}_blocks_6_5_4_score=0.75.jpg')
exit(0)