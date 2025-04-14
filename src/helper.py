import pandas as pd
from torchvision import datasets, transforms
from timm.data import create_transform
import PIL
from util.crop import GridCropAndResize

def load_class_mapping(class_list_file):
    with open(class_list_file) as f:
        class_index_to_class_name = {i: line.strip() for i, line in enumerate(f)}
    return class_index_to_class_name


def load_species_mapping(species_map_file):
    df = pd.read_csv(species_map_file, sep=';', quoting=1, dtype={'species_id': str})
    df = df.set_index('species_id')
    return  df['species'].to_dict()


def build_test_transform(data_config, n=None):
    #transform = transforms.Compose([
    #    transforms.Resize(518, interpolation=PIL.Image.InterpolationMode.BICUBIC),  # Ensures the shorter side = 518
    #    transforms.CenterCrop(518* data_config.crop_pct),  # No-op when crop_pct = 1.0, just keeps size 518x518
    #    transforms.ToTensor(),  # Converts PIL image to PyTorch Tensor (C, H, W) in [0, 1]
    #    transforms.Normalize(mean=data_config.mean, std=data_config.std),  # ImageNet normalization
    #])

    # Transform above does resize then crop. It keeps the aspect ratio but may cut parts of the image. 
    # The transform below may distort the image, as aspect ratio may not be preserved but the content remains. 
    transform = transforms.Compose([
        transforms.Resize((2816,2816), interpolation=PIL.Image.BICUBIC),  # Optional: scale to be divisible by 16
        transforms.ToTensor(),  # Convert to tensor: (C, H, W)
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
        GridCropAndResize(crop_size=n),  # Now that it's a tensor, we can safely patchify
    ])


    return transform

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
    