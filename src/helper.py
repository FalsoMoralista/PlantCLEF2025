import pandas as pd
from torchvision import datasets, transforms
from timm.data import create_transform
import PIL
from util.crop import GridCropAndResize
import sys
import logging
from src.utils.schedulers import (WarmupCosineSchedule, CosineWDSchedule)
import torch
from torch import inf

from PIL import Image

import torch.nn.functional as F
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt
import numpy as np
import os

from torchvision.utils import make_grid

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()

def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()


def print_tensor_size(tensor):
    bytes_per_element = tensor.element_size()
    
    # Get total number of elements
    num_elements = tensor.nelement()
    
    # Calculate total bytes
    memory_bytes = bytes_per_element * num_elements
    memory_mb = memory_bytes / (1024 * 1024)
    memory_gb = memory_mb / 1024
    
    print(f"Tensor shape: {tensor.shape}")
    print(f"Data type: {tensor.dtype}")
    print(f"Size per element: {bytes_per_element} bytes")
    print(f"Number of elements: {num_elements:,}")
    print(f"Memory usage: {memory_bytes:,} bytes ({memory_mb:.2f} MB, {memory_gb:.2f} GB)")    
    

# Borrowed from MAE.
class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, retain_graph=None, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm

def init_opt(
    encoder,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False,
    ipe_scale=1.25
):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0 
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(encoder.parameters())

    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(ipe_scale*num_epochs*iterations_per_epoch))
    scaler = NativeScalerWithGradNormCount() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler 


def visualize_global_attention_on_image(image, attn_map, patch_size=64, save_path=None):
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


def visualize_clustered_crops(selected_crops, cluster_assignments, image_name, save_path=None):
    """
    Visualize and group selected crops by cluster assignment.

    Args:
        selected_crops (Tensor): (N, 3, 518, 518) crop images.
        cluster_assignments (Tensor): (N,) cluster ids.
        image_name (str): Original image name (for figure title or saving).
        save_path (str): Optional path to save figure.
    """
    num_clusters = cluster_assignments.max().item() + 1
    to_pil = transforms.ToPILImage()
    
    fig, axes = plt.subplots(num_clusters, 1, figsize=(20, num_clusters * 4))
    if num_clusters == 1:
        axes = [axes]
    
    for cluster_id in range(num_clusters):
        cluster_idxs = (cluster_assignments == cluster_id).nonzero(as_tuple=True)[0]
        cluster_crops = selected_crops[cluster_idxs]
        
        # Take up to 10 images per cluster for visualization
        cluster_crops = cluster_crops[:10]
        grid = make_grid(cluster_crops, nrow=5, padding=2, normalize=True)
        axes[cluster_id].imshow(grid.permute(1, 2, 0).cpu())
        axes[cluster_id].axis("off")
        axes[cluster_id].set_title(f"Cluster {cluster_id} ({len(cluster_idxs)} crops)", fontsize=16)

    fig.suptitle(f"Crop Clustering for {image_name}", fontsize=18)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f"{save_path}/{image_name}_clusters.png", bbox_inches='tight')
    else:
        plt.show()


def get_high_attention_crop_positions(attn_scores, crop_size, image_size, attn_shape=[32,32],  threshold=0.2):
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

    # 1. Average over heads
    attn_scores = attn_scores.mean(dim=1)[0]  # Shape: [1024, 1024]

    # 2. Aggregate attention received by each patch (i.e., sum over rows → what each token receives)
    attn_scores = attn_scores.sum(dim=0)  # [1024]

    # 3. Min max normalization
    eps = 1e-6
    attn_scores = (attn_scores - attn_scores.min()) / (attn_scores.max() - attn_scores.min() + eps)
    attn_scores = attn_scores.reshape(attn_shape[0], attn_shape[1])

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


def crop_at_positions(crops, positions, crop_size=64, image_size=2048):
    """
    Selects pre-resized crop patches from a 2048x2048 image based on (x, y) positions.

    Args:
        crops (Tensor): Tensor of shape (1024, 3, 518, 518) representing pre-resized 64x64 crops.
        positions (Tensor): Tensor of shape (N, 2), (x, y) top-left positions in the original image space.
        crop_size (int): Original crop size (default: 64).
        image_size (int): Original image size (default: 2048).

    Returns:
        Tensor of selected crops (N, 3, 518, 518)
    """
    grid_size = image_size // crop_size  # 2048 // 64 = 32

    selected = []
    for x, y in positions:
        x_idx = int(x) // crop_size
        y_idx = int(y) // crop_size
        flat_idx = y_idx * grid_size + x_idx

        if 0 <= flat_idx < crops.shape[0]:
            selected.append(crops[flat_idx])
    
    if not selected:
        return None

    return torch.stack(selected)

def non_square_crop_at_positions(crops, positions, crop_size=64, image_height=3072, image_width=2048):
    """
    Selects pre-resized crop patches from an image of shape (image_height, image_width)
    based on (x, y) positions.

    Args:
        crops (Tensor): Tensor of shape (Hc * Wc, 3, H, W) representing pre-resized crop patches.
        positions (Tensor): Tensor of shape (N, 2), (x, y) top-left positions in the original image space.
        crop_size (int): Original crop size (default: 64).
        image_height (int): Height of the original image (default: 3072).
        image_width (int): Width of the original image (default: 2048).

    Returns:
        Tensor of selected crops (N, 3, H, W)
    """
    grid_height = image_height // crop_size  # 3072 // 64 = 48
    grid_width = image_width // crop_size    # 2048 // 64 = 32

    selected = []
    for x, y in positions:
        x_idx = int(x) // crop_size
        y_idx = int(y) // crop_size
        flat_idx = y_idx * grid_width + x_idx

        if 0 <= x_idx < grid_width and 0 <= y_idx < grid_height and 0 <= flat_idx < crops.shape[0]:
            selected.append(crops[flat_idx])
    
    if not selected:
        return None

    return torch.stack(selected)


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



def visualize_patch_grids(patch_crops_list, K, save_dir, base_filename='high_attn_patch',
                          mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Save each KxK patch grid as an image, denormalizing if needed.

    Args:
        patch_crops_list (list of tensors): Each tensor is (K*K, 3, H, W)
        K (int): Grid size
        save_dir (str): Directory to save images
        base_filename (str): Base filename to use
        mean, std (list): Normalization values used on input tensors
    """
    os.makedirs(save_dir, exist_ok=True)

    def denormalize(tensor, mean, std):
        """
        Denormalizes a batch of tensors (N, C, H, W) or a single tensor (C, H, W)
        using the given mean and std.
        """
        mean = torch.tensor(mean).view(-1, 1, 1).to(tensor.device)
        std = torch.tensor(std).view(-1, 1, 1).to(tensor.device)
        return tensor * std + mean

    for idx, crop_tensor in enumerate(patch_crops_list):
        crop_tensor = denormalize(crop_tensor, mean, std).clamp(0, 1)
        grid = make_grid(crop_tensor, nrow=K, padding=2, pad_value=0.0)  # 1-> white padding 0-> black
        img = TF.to_pil_image(grid.cpu())
        img.save(os.path.join(save_dir, f"{base_filename}_{idx:03d}.png"))


def get_kxk_neighborhood_patches(high_attn_positions, K, patch_grid_shape, img_tensor):
    """
    For each high-attention position (index in patch grid), return a KxK neighborhood of patches.

    Args:
        high_attn_positions (Tensor): Pixel positions of high attention patches (x, y).
        K (int): Neighborhood size (must be odd).
        patch_grid_shape (tuple): (H_patches, W_patches), e.g., (48, 32)
        img_tensor (Tensor): Shape (1536, 3, 64, 64)

    Returns:
        list of tensors: Each element is a tensor of shape (K*K, 3, 64, 64)
    """
    assert K % 2 == 1, "K must be an odd number"
    
    H_patches, W_patches = patch_grid_shape
    patch_crops = []

    # Precompute index grid
    for x_pix, y_pix in high_attn_positions:
        i = y_pix.item() // 64  # row
        j = x_pix.item() // 64  # col

        neighbors = []

        for dy in range(-(K // 2), K // 2 + 1):
            for dx in range(-(K // 2), K // 2 + 1):
                ni = i + dy
                nj = j + dx
                if 0 <= ni < H_patches and 0 <= nj < W_patches:
                    idx = ni * W_patches + nj
                    neighbors.append(img_tensor[idx])
                else:
                    # If outside bounds, pad with black image
                    neighbors.append(torch.zeros_like(img_tensor[0]))

        patch_crops.append(torch.stack(neighbors))  # (K*K, 3, 64, 64)

    return patch_crops
