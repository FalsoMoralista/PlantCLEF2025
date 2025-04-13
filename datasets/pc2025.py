import timm
import torch
import torchvision
import PIL
import os
from torchvision import datasets, transforms
from timm.data import create_transform

from util.patchify import Patchify

def build_dataset(is_train, image_folder, transform):
    root = os.path.join(image_folder, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)
    return dataset

# Average width is 2776 (+- 414), average height is 2769 (+- 362) 
# therefore we will resize to a 2816 square image 
def build_transform(is_training, data_config, N=None):
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=PIL.Image.InterpolationMode.BICUBIC),  # Ensures the shorter side = 518
        transforms.CenterCrop(518* data_config.crop_pct),  # No-op when crop_pct = 1.0, just keeps size 518x518
        transforms.ToTensor(),  # Converts PIL image to PyTorch Tensor (C, H, W) in [0, 1]
        transforms.Normalize(mean=data_config.mean, std=data_config.std),  # ImageNet normalization
    ])

    # TODO: test this:
    transform = transforms.Compose([
        transforms.Resize(2816, interpolation=PIL.Image.BICUBIC),  # Optional: scale to be divisible by 16
        transforms.CenterCrop(2816),  # Optional: crop square
        transforms.ToTensor(),  # Convert to tensor: (C, H, W)
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        Patchify(n=16),  # Now that it's a tensor, we can safely patchify
    ])
    return transform


def build_test_dataset(image_folder,data_config, is_training, batch_size, N=None):
    
    transform = build_transform(is_training, data_config)

    dataset = build_dataset(is_train=is_training, transform=transform, image_folder=image_folder) 
    
    print('Test dataset created')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=1,
        rank=0)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=None,
        sampler=dist_sampler,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=True,
        num_workers=8,
        persistent_workers=False)    
    return dataset, data_loader, dist_sampler

