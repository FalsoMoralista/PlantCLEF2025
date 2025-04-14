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
    