import timm
import torch
import torchvision
import PIL
import os
from torchvision import datasets, transforms
from timm.data import create_transform

from util.crop import GridCropAndResize

class CustomImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super(CustomImageFolder, self).__init__(root, transform)
        
    def __getitem__(self, index):
        # Get image and label using the parent method
        img, label = super(CustomImageFolder, self).__getitem__(index)
        
        # Get the path and extract the filename
        path = self.samples[index][0]
        filename = os.path.basename(path)
        
        # Return image, label, and filename
        return img, label, filename


def build_test_transform(data_config, input_resolution=(2048,2048), n=None):
    # Transform by doing resize then crop may keep the aspect ratio but may cut parts of the image. 
    # The transform below may distort the image, as aspect ratio may not be preserved but the content remains. 
    transform = transforms.Compose([
        transforms.Resize(input_resolution, interpolation=PIL.Image.BICUBIC),  # Optional: scale to be divisible by 16
        transforms.ToTensor(),  # Convert to tensor: (C, H, W)
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        GridCropAndResize(crop_size=n),
    ])

    return transform

def build_inference_transform(data_config, input_resolution=(2048,2048), n=None):
    # Transform by doing resize then crop may keep the aspect ratio but may cut parts of the image. 
    # The transform below may distort the image, as aspect ratio may not be preserved but the content remains. 
    transform = transforms.Compose([
        transforms.Resize(input_resolution, interpolation=PIL.Image.BICUBIC),  # Optional: scale to be divisible by 16
        transforms.ToTensor(),  # Convert to tensor: (C, H, W)
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    return transform

def build_train_transform(data_config):
    transform = transforms.Compose([
        transforms.Resize(518, interpolation=PIL.Image.BICUBIC),  # Optional: scale to be divisible by 16
        transforms.CenterCrop(518),
        transforms.ToTensor(),  # Convert to tensor: (C, H, W)
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    return transform


def build_train_dataset(image_folder, world_size, rank, data_config, batch_size=2048, num_workers=8):
    
    transform = build_train_transform(data_config)

    dataset = CustomImageFolder(image_folder, transform=transform)
    
    print(f'Train dataset created for rank {rank},/{world_size}')

    dist_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=None,
        sampler=dist_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
        shuffle=False,
        persistent_workers=False) 
    print('Data Loader length:', len(data_loader), f'process_id: {rank}')   
    return dataset, data_loader, dist_sampler

def build_test_dataset(image_folder, data_config, input_resolution=(2048,2048), batch_size=1, num_workers=16, n=None, world_size=0, rank=None, shuffle=False):
    
    transform = build_test_transform(data_config, input_resolution=input_resolution ,n=n)

    dataset = CustomImageFolder(image_folder, transform=transform)
    
    print('Test dataset created for rank', rank, 'world_size: ', world_size)
    if world_size > 1:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            shuffle=shuffle,
            rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=None,
        sampler=dist_sampler if world_size > 1 else None,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=num_workers)    
    return dataset, data_loader, dist_sampler if world_size > 1 else None

def build_inference_dataset(image_folder, data_config, input_resolution=(2048,2048), batch_size=1, num_workers=16, n=None, world_size=0, rank=None, shuffle=False):
    
    transform = build_inference_transform(data_config, input_resolution=input_resolution ,n=n)

    dataset = CustomImageFolder(image_folder, transform=transform)
    
    print('Test dataset created for rank', rank, 'world_size: ', world_size)
    if world_size > 1:
        dist_sampler = torch.utils.data.distributed.DistributedSampler(
            dataset=dataset,
            num_replicas=world_size,
            shuffle=shuffle,
            rank=rank)
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=None,
        sampler=dist_sampler if world_size > 1 else None,
        batch_size=batch_size,
        drop_last=False,
        pin_memory=False,
        num_workers=num_workers)    
    return dataset, data_loader, dist_sampler if world_size > 1 else None

