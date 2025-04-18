import sys
sys.path.append('../')
import torch
import os
os.environ['TORCH_DEBUG'] = 'DETAIL'
import timm

from src.helper import load_class_mapping
from datasets.pc2025 import build_test_dataset

#from src.models.ijepa.vision_transformer import VisionTransformer
from src.models.custom_vision_transformer import VisionTransformer


pretrained_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar'
cid_to_spid = load_class_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/class_mapping.txt')
test_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/test_dataset/'
patch_size = 32
batch_size = 2
if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)


model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction

data_config = timm.data.resolve_model_data_config(model)


test_dataset, test_dataloader = build_test_dataset(image_folder=test_dir,
                                            data_config=data_config,
                                            num_workers=4,
                                            n=patch_size,
                                            batch_size=batch_size)

ViT = VisionTransformer(img_size=[2048,1024], pretrained_patch_embedder=model,patch_size=patch_size, embed_dim=768,depth=6)
ViT.to(device)
print('Loading Vision Transformer:', ViT)
print('Allocated Memory with model loading:', (torch.cuda.memory_allocated() / 1024.**3), ' GB')

criterion = torch.nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()
optimizer = torch.torch.optim.AdamW(ViT.parameters())

for im_id, (preprocessed_images, _, name) in enumerate(test_dataloader):
    x = preprocessed_images.to(device)
    y = torch.randn(batch_size, 768).to(device)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
        output = ViT(x)
        output = torch.mean(output, dim=1).squeeze(1) # alternative
        print('output:', output.size())
        loss = criterion(output, y)
        print('Allocated Memory with loss computation:', (torch.cuda.memory_allocated() / 1024.**3), ' GB')
    # -- Deals with mixed precision
    scaler.scale(loss).backward(create_graph=False, retain_graph=False) 
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()        
    # -- 
    print('Max Memory Allocated so far [mem: %.2e] :' %(torch.cuda.max_memory_allocated() / 1024.**3), ' GB')
    break

