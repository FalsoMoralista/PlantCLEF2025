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
    def __init__(self, crop_size, output_size=518, return_positions=False):
        super().__init__()
        self.crop_size = crop_size
        self.output_size = output_size
        self.return_positions = return_positions
        
    def forward(self, img):
        """
        Args:
            img: Tensor in (C, H, W) format
            
        Returns:
            Tensor of stacked crops, shape (N, C, output_size, output_size) where N is number of crops
            Or tuple of (crops, positions) if return_positions is True
        """
        # Get tensor dimensions
        c, height, width = img.shape
        
        # Calculate number of crops in each dimension
        num_crops_y = height // self.crop_size
        num_crops_x = width // self.crop_size
        total_crops = num_crops_y * num_crops_x
        
        # Create position indices if needed
        if self.return_positions:
            # Create grid of x, y positions
            x_indices = torch.arange(0, num_crops_x, device=img.device) * self.crop_size
            y_indices = torch.arange(0, num_crops_y, device=img.device) * self.crop_size
            
            # Create meshgrid of all positions
            y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')
            positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        # Handle the case where crop_size equals output_size (no resize needed)
        if self.crop_size == self.output_size:
            # Use unfold for efficient cropping
            crops = img.unfold(1, self.crop_size, self.crop_size)  # Unfold height dimension
            crops = crops.unfold(2, self.crop_size, self.crop_size)  # Unfold width dimension
            
            # Reshape to (N, C, crop_size, crop_size)
            crops = crops.permute(1, 2, 0, 3, 4).reshape(total_crops, c, self.crop_size, self.crop_size)
            
            if self.return_positions:
                return crops, positions
            return crops
        else:
            # For the resize case, use unfold for cropping
            crops = img.unfold(1, self.crop_size, self.crop_size)  # Unfold height dimension
            crops = crops.unfold(2, self.crop_size, self.crop_size)  # Unfold width dimension
            
            # Reshape to (N, C, crop_size, crop_size)
            crops = crops.permute(1, 2, 0, 3, 4).reshape(total_crops, c, self.crop_size, self.crop_size)
            
            # Batch resize all crops at once
            resized_crops = F.interpolate(
                crops,
                size=(self.output_size, self.output_size),
                mode='bilinear',
                align_corners=False,
                antialias=True
            )
            
            if self.return_positions:
                return resized_crops, positions
            return resized_crops

class GridCrop(nn.Module):
    """
    Crops a tensor image into a grid of non-overlapping squares and resizes each crop
    to a fixed size for model processing.
    
    Args:
        crop_size (int): Size of each square crop (n x n pixels
        return_positions (bool): If True, returns crop positions along with tensors
    """
    def __init__(self, crop_size, return_positions=False):
        super().__init__()
        self.crop_size = crop_size
        self.return_positions = return_positions
        
    def forward(self, img):
        """
        Args:
            img: Tensor in (C, H, W) format
            
        Returns:
            Tensor of stacked crops, shape (N, C, output_size, output_size) where N is number of crops
            Or tuple of (crops, positions) if return_positions is True
        """
        # Get tensor dimensions
        c, height, width = img.shape
        
        # Calculate number of crops in each dimension
        num_crops_y = height // self.crop_size
        num_crops_x = width // self.crop_size
        total_crops = num_crops_y * num_crops_x
        
        # Create position indices if needed
        if self.return_positions:
            # Create grid of x, y positions
            x_indices = torch.arange(0, num_crops_x, device=img.device) * self.crop_size
            y_indices = torch.arange(0, num_crops_y, device=img.device) * self.crop_size
            
            # Create meshgrid of all positions
            y_grid, x_grid = torch.meshgrid(y_indices, x_indices, indexing='ij')
            positions = torch.stack([x_grid.flatten(), y_grid.flatten()], dim=1)
        
        crops = img.unfold(1, self.crop_size, self.crop_size)  # Unfold height dimension
        crops = crops.unfold(2, self.crop_size, self.crop_size)  # Unfold width dimension
        
        # Reshape to (N, C, crop_size, crop_size)
        crops = crops.permute(1, 2, 0, 3, 4).reshape(total_crops, c, self.crop_size, self.crop_size)
                    
        if self.return_positions:
            return crops, positions
        return crops

