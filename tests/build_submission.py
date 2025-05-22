import sys
sys.path.append('../')
import torch
import os
os.environ['TORCH_DEBUG'] = 'DETAIL'
import timm

from src.helper import load_class_mapping, load_species_mapping
from datasets.pc2025 import build_test_dataset, build_test_transform


import faiss
import faiss.contrib.torch_utils

from src.models.custom_vision_transformer import PilotVisionTransformer
from PIL import Image
import random
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torchvision.utils import make_grid

from src.k_means import KMeansModule, cosine_cluster_index

from src.helper import(visualize_global_attention_on_image,
                       mask_image_from_attention,
                       visualize_clustered_crops,
                       get_high_attention_crop_positions,
                       crop_at_positions,
                       non_square_crop_at_positions,
                       plot_crops_in_grid_positions,
                       get_kxk_neighborhood_patches,
                       plot_and_save_stacked_image,
                       visualize_patch_grids)

import csv

pretrained_path = '/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all/model_best.pth.tar'
cid_to_spid = load_class_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/class_mapping.txt')
spid_to_sp = load_species_mapping('/home/rtcalumby/adam/luciano/PlantCLEF2025/PlantCLEF2025/pretrained_models/species_id_to_name.txt')
test_dir = '/home/rtcalumby/adam/luciano/PlantCLEF2025/test_dataset/'
patch_size = 64 # TODO: modify
batch_size = 1

if not torch.cuda.is_available():
    device = torch.device('cpu')
else:
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

images = os.listdir(test_dir+'test/')
print('len images:', len(images))

epochs = [10] # TODO: modify

model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
model.to(device)
#cls_head = model.head
model.head = torch.nn.Identity() # Replace classification head by identity layer for feature extraction

dino_cls = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False, num_classes=len(cid_to_spid), checkpoint_path=pretrained_path)
dino_cls.to(device)

data_config = timm.data.resolve_model_data_config(model)

experiment_code = 2
input_resolution=(3072,2048)
attn_shape = (48,32)

test_dataset, test_dataloader, dist_sampler = build_test_dataset(image_folder=test_dir,
                                            data_config=data_config,
                                            input_resolution=input_resolution,
                                            num_workers=12,
                                            n=64,
                                            world_size=1,
                                            rank=0,
                                            shuffle=False,
                                            batch_size=1)

r_path = f'/home/rtcalumby/adam/luciano/PlantCLEF2025/logs/experiment_{experiment_code}/'

resources = faiss.StandardGpuResources()
config = faiss.GpuIndexFlatConfig()
config.device = 0

threshold = 0.6
for epoch_no in epochs:
    checkpoint = torch.load(r_path+f"experiment_{experiment_code}-ep{epoch_no}.pth.tar", map_location=torch.device('cpu'))
    ViT = PilotVisionTransformer(img_size=[input_resolution[0],input_resolution[1]], pretrained_patch_embedder=model,patch_size=patch_size, embed_dim=768, depth=6)
    print('Loading Vision Transformer:', ViT)        

    pretrained_dict = checkpoint['custom_vit']
    msg = ViT.load_state_dict(pretrained_dict)
    print('Loading state_dict:', msg)
    ViT.linear = torch.nn.Identity()
    ViT.to(device)
    
    logit_list = {}
    clustering = False
    for itr, (img_tensor, _, image_name) in enumerate(test_dataloader):
        if itr % 50 == 0:
            print(f'Iteration:{100*((itr+1)/2106)} %', flush=True)
            print('Max mem allocated:', torch.cuda.max_memory_allocated() / 1024.**3)        
        
        img_tensor = img_tensor.squeeze(0)
        # Batch resize all crops at once
        resized_crops = F.interpolate( # Shape (1536, 3, 518, 518)
            img_tensor,
            size=(518, 518),
            mode='bilinear',
            align_corners=False,
            antialias=True
        ).to(device)
        
        image_name = image_name[0]
        img_tensor = img_tensor.to(device)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):        
            with torch.inference_mode():
                _, attn = ViT(resized_crops, return_attention=True)
                del resized_crops

        num_blocks = 6
        attn_map = torch.stack([attn[i][1] for i in range(num_blocks)]).mean(0)

        high_attn_positions = get_high_attention_crop_positions(
            attn_scores=attn_map,
            crop_size=64,
            attn_shape=attn_shape,
            image_size=input_resolution,
            threshold=threshold
        )        

        K = 5
        patch_crops_list = get_kxk_neighborhood_patches(
            high_attn_positions=high_attn_positions,
            K=K,
            patch_grid_shape=attn_shape,
            img_tensor=img_tensor
        )

        del attn
        del attn_map
        del img_tensor
        torch.cuda.empty_cache()

        # Reshape (build image from a stack of neighbouring patches)
        patch_crops_list = torch.stack(patch_crops_list).to('cpu')
        B, _, C, H, W = patch_crops_list.shape
        patch_crops_list = patch_crops_list.view(B, K, K, C, H, W)
        patch_crops_list = patch_crops_list.permute(0, 3, 1, 4, 2, 5)  # (B, C, K, H, K, W)
        patch_crops_list = patch_crops_list.reshape(B, C, K * H, K * W)  
        
        resized_crop_list = F.interpolate( # Shape (B, 3, 64 * K, 64 * K)
            patch_crops_list,
            size=(518, 518),
            mode='bilinear',
            align_corners=False,
            antialias=True
        ).to(device) # out -> (B, C, 518, 518)
        del patch_crops_list

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
            with torch.inference_mode():                  
                out = dino_cls(resized_crop_list).detach().cpu()

        if not clustering:
            logit_list[image_name.replace('.jpg', '')] = out
        else: 
            image_predictions = []
            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=True):
                with torch.inference_mode():
                    stacked_crops_embeddings = model(resized_crop_list)
            print('Selected crop embeddings size:', stacked_crops_embeddings.size())
            
            k_range = [2,3,4,5,6,7,8,9,10]
            n_kmeans = [KMeansModule(K=k, dimensionality=768, config=config, resources=resources) for k in k_range]

            for k_means in n_kmeans:
                energy = k_means.train_from_batch(x=stacked_crops_embeddings.cpu())
                
            best_K, cluster_assignments = cosine_cluster_index(n_kmeans=n_kmeans,
                                                                        k_range=k_range,
                                                                        xb=stacked_crops_embeddings,
                                                                        device=device)
            print(f'Best K: {best_K+2}, cls_assign: {cluster_assignments.size()}')
            print('Cls assignments:', cluster_assignments)
            print('out:', out.size())

            predictions_by_cluster = [[] for _ in range(best_K+2)]
            for idx, item in enumerate(cluster_assignments):
                best_K_idx = item.item()
                predictions_by_cluster[best_K_idx].append(out[idx])
            
            for k in range(best_K+2):
                predictions_by_cluster[k] = torch.stack(predictions_by_cluster[k])
            print('length Predictions:', len(predictions_by_cluster))
            print('Predictions:', predictions_by_cluster[0].size())
                
            logit_list[image_name.replace('.jpg', '')] = predictions_by_cluster
            visualize_clustered_crops(selected_crops=resized_crop_list.cpu(), cluster_assignments=cluster_assignments.cpu(), image_name=image_name,save_path='attention/pilot')       

    print('predicted items length:', len(logit_list))
    if not clustering:
        torch.save(logit_list, f'logits/stack_{K}_predictions_run_{experiment_code}_ep{epoch_no}_treshold={threshold}.pt')
    else:
        torch.save(logit_list, f'logits/clus+stack_{K}_predictions_run_{experiment_code}_ep{epoch_no}_treshold={threshold}.pt')
