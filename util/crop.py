import torch
import torch.nn as nn
import torch.nn.functional as F

class GridCropAndResize(nn.Module):
    """
    Crops a tensor image into a grid of non-overlapping squares and resizes each crop
    to a fixed size for model processing.
    
    Args:
        crop_size (int): Size of each square crop (n x n pixels)
        output_size (int): Size to resize each crop to (default: 518)
        return_positions (bool): If True, returns crop positions along with tensors
    """
    def __init__(self, crop_size, output_size=518, return_positions = False):
        super().__init__()
        self.crop_size = crop_size
        self.output_size = output_size
        self.return_positions = return_positions
        
    def forward(self, img):
        """
        Args:
            img: Tensor in (C, H, W) format
            
        Returns:
            List of cropped and resized tensors, each of shape (C, output_size, output_size)
        """
        # Get tensor dimensions
        c, height, width = img.shape
        
        # Calculate number of crops in each dimension
        num_crops_x = width // self.crop_size
        num_crops_y = height // self.crop_size
        
        # Create crops
        crops = []
        positions = []
        
        for y in range(num_crops_y):
            for x in range(num_crops_x):
                # Calculate crop coordinates
                left = x * self.crop_size
                upper = y * self.crop_size
                
                # Extract the crop from the tensor
                crop = img[:, upper:upper+self.crop_size, left:left+self.crop_size]
                
                # Resize using interpolation
                if self.crop_size != self.output_size:
                    resized_crop = F.interpolate(
                        crop.unsqueeze(0),  # Add batch dimension for interpolate
                        size=(self.output_size, self.output_size),
                        mode='bilinear',
                        align_corners=False,
                        antialias=True
                    ).squeeze(0)  # Remove batch dimension
                else:
                    resized_crop = crop
                
                crops.append(resized_crop)
                
                if self.return_positions:
                    positions.append((left, upper))
        
        if self.return_positions:
            return crops, positions
        
        return torch.stack(crops)