import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import math
from matplotlib.colors import LinearSegmentedColormap

class AttentionMapVisualizer:
    """
    Visualizes attention maps from a Vision Transformer for high-resolution images
    that have been processed in patches.
    """
    def __init__(self, model, device=None, patch_size=64, img_size=(2048, 2048)):
        """
        Initialize the visualizer.
        
        Args:
            model: The PilotVisionTransformer model
            device: The device to run computations on
            patch_size: Size of each patch
            img_size: Size of the original image (height, width)
        """
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        self.patch_size = patch_size
        self.img_size = img_size
        
        # Number of patches in each dimension
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        
        # Register hooks
        self.attention_maps = []
        self.hooks = []
        self._register_hooks()
        
    def _register_hooks(self):
        """Register forward hooks to capture attention maps from all transformer blocks."""
        def get_attention(name):
            def hook(module, input, output):
                # Extract attention weights
                # This assumes the output of the attention module includes attention weights
                # You may need to adjust this based on your specific implementation
                attn = module.attn.attn.detach().clone()  # Shape: [batch_size, num_heads, seq_len, seq_len]
                self.attention_maps.append((name, attn))
            return hook
            
        # Register hooks for each transformer block
        for i, block in enumerate(self.model.blocks):
            self.hooks.append(block.register_forward_hook(get_attention(f"block_{i}")))
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def visualize_attention(self, image_path, img_tensor, layer_idx=-1, head_idx=None, 
                           save_path=None, alpha=0.6, cmap='viridis'):
        """
        Visualize attention maps for a given image.
        
        Args:
            image_path: Path to the input image
            transform: Transform to apply to the image
            layer_idx: Index of the transformer layer to visualize (-1 for last layer)
            head_idx: Index of the attention head to visualize (None to average all heads)
            save_path: Path to save the visualization
            alpha: Transparency of the attention map overlay
            cmap: Colormap for the attention visualization
            
        Returns:
            The visualization figure
        """
        # Clear previous attention maps
        self.attention_maps = []

        
        # Get original image for visualization
        img_for_display = Image.open(image_path).convert('RGB')
        img_for_display = img_for_display.resize(self.img_size, Image.BICUBIC)
        
        # Process through the model
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # Get attention map from the specified layer
        layer_name, attn_map = self.attention_maps[layer_idx]
        
        # Average across heads if head_idx is None, otherwise select the specified head
        if head_idx is None:
            # Average over all heads
            attn_map = attn_map.mean(1)  # Shape: [batch_size, seq_len, seq_len]
        else:
            # Select specific head
            attn_map = attn_map[:, head_idx]  # Shape: [batch_size, seq_len, seq_len]
        
        # For a grid visualization, we want to see the attention paid by each patch
        # to all other patches. We can visualize this by taking the attention weights
        # and reshaping them to match our grid of patches.
        
        # Get the number of patches
        num_patches = self.num_patches_h * self.num_patches_w
        
        # Create a figure
        fig, axs = plt.subplots(self.num_patches_h, self.num_patches_w, 
                                figsize=(20, 20), 
                                gridspec_kw={'wspace': 0, 'hspace': 0})
        
        # Get the original image
        img_np = np.array(img_for_display)
        
        # Create custom colormap
        attention_cmap = plt.get_cmap(cmap)
        
        # For each patch position
        for i in range(self.num_patches_h):
            for j in range(self.num_patches_w):
                # Calculate the patch index
                patch_idx = i * self.num_patches_w + j
                
                # Get attention scores for this patch
                # Shape of attn_map: [batch_size, num_patches, num_patches]
                # We want the attention from this patch to all others
                attention_scores = attn_map[0, patch_idx].reshape(self.num_patches_h, self.num_patches_w)
                
                # Normalize attention scores for better visualization
                attention_scores = (attention_scores - attention_scores.min()) / (attention_scores.max() - attention_scores.min() + 1e-8)
                
                # Create a heatmap
                if self.num_patches_h == 1 and self.num_patches_w == 1:
                    ax = axs
                else:
                    ax = axs[i, j] if self.num_patches_h > 1 and self.num_patches_w > 1 else axs[max(i, j)]
                
                # Show the image
                ax.imshow(img_np)
                
                # Create an attention overlay
                attention_overlay = np.zeros((self.img_size[0], self.img_size[1], 4))
                
                # Fill the attention overlay
                for ni in range(self.num_patches_h):
                    for nj in range(self.num_patches_w):
                        # Get the attention score for this target patch
                        score = attention_scores[ni, nj].item()
                        
                        # Calculate the pixel coordinates
                        y_start = ni * self.patch_size
                        y_end = (ni + 1) * self.patch_size
                        x_start = nj * self.patch_size
                        x_end = (nj + 1) * self.patch_size
                        
                        # Get the color based on score
                        color = attention_cmap(score)
                        
                        # Set the overlay with the attention color and alpha
                        attention_overlay[y_start:y_end, x_start:x_end] = [color[0], color[1], color[2], alpha * score]
                
                # Overlay the attention map
                ax.imshow(attention_overlay)
                
                # Highlight the current patch
                patch_y_start = i * self.patch_size
                patch_y_end = (i + 1) * self.patch_size
                patch_x_start = j * self.patch_size
                patch_x_end = (j + 1) * self.patch_size
                
                # Draw a box around the current patch
                rect = plt.Rectangle((patch_x_start, patch_y_start),
                                      self.patch_size, self.patch_size,
                                      edgecolor='red', facecolor='none', linewidth=2)
                ax.add_patch(rect)
                
                # Remove axis
                ax.axis('off')
        
        plt.suptitle(f"Attention Map for Layer {layer_idx} {'(Average of all heads)' if head_idx is None else f'(Head {head_idx})'}", 
                     fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig
    
    def visualize_average_attention(self, image_path, img_tensor, layer_idx=-1, head_idx=None,
                                   save_path=None, cmap='hot'):
        """
        Visualize the average attention map across all patches.
        
        Args:
            image_path: Path to the input image
            transform: Transform to apply to the image
            layer_idx: Index of the transformer layer to visualize (-1 for last layer)
            head_idx: Index of the attention head to visualize (None to average all heads)
            save_path: Path to save the visualization
            cmap: Colormap for the attention visualization
            
        Returns:
            The visualization figure
        """
        # Clear previous attention maps
        self.attention_maps = []
        
    
        # Get original image for visualization
        img_for_display = Image.open(image_path).convert('RGB')
        img_for_display = img_for_display.resize(self.img_size, Image.BICUBIC)
        
        # Process through the model
        with torch.no_grad():
            _ = self.model(img_tensor)
        
        # Get attention map from the specified layer
        layer_name, attn_map = self.attention_maps[layer_idx]
        
        # Average across heads if head_idx is None, otherwise select the specified head
        if head_idx is None:
            # Average over all heads
            attn_map = attn_map.mean(1)  # Shape: [batch_size, seq_len, seq_len]
        else:
            # Select specific head
            attn_map = attn_map[:, head_idx]  # Shape: [batch_size, seq_len, seq_len]
        
        # Average attention across all source patches to get an overall attention heatmap
        # This shows which patches are generally attended to the most across the image
        avg_attn = attn_map.mean(1)[0]  # Shape: [num_patches]
        
        # Reshape to match the grid layout
        avg_attn = avg_attn.reshape(self.num_patches_h, self.num_patches_w)
        
        # Normalize
        avg_attn = (avg_attn - avg_attn.min()) / (avg_attn.max() - avg_attn.min() + 1e-8)
        
        # Create figure
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        
        # Show original image
        ax[0].imshow(img_for_display)
        ax[0].set_title("Original Image", fontsize=14)
        ax[0].axis('off')
        
        # Create upsampled attention map
        attn_map_upsampled = F.interpolate(
            avg_attn.unsqueeze(0).unsqueeze(0), 
            size=self.img_size,
            mode='bilinear',
            align_corners=False
        ).squeeze().cpu().numpy()
        
        # Show attention heatmap
        img_np = np.array(img_for_display)
        ax[1].imshow(img_np)
        heatmap = ax[1].imshow(attn_map_upsampled, alpha=0.6, cmap=cmap)
        ax[1].set_title(f"Average Attention for Layer {layer_idx} {'(All heads)' if head_idx is None else f'(Head {head_idx})'}", 
                        fontsize=14)
        ax[1].axis('off')
        
        # Add colorbar
        plt.colorbar(heatmap, ax=ax[1], shrink=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        return fig


# Usage example:
def visualize_attention_maps(model_path, image_path, data_config, patch_size=64, img_size=(2048, 2048)):
    """
    Example function to visualize attention maps for a given model and image.
    
    Args:
        model_path: Path to the model checkpoint
        image_path: Path to the image
        data_config: Configuration for data preprocessing
        patch_size: Size of each patch
        img_size: Size of the input image
    """
    import timm
    import torch
    from torch import nn
    
    # Load DINOv2 model
    dino_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False)
    dino_model.head = nn.Identity()  # Replace classification head with identity
    
    # Create PilotVisionTransformer
    model = PilotVisionTransformer(
        img_size=img_size,
        pretrained_patch_embedder=dino_model,
        patch_size=patch_size,
        embed_dim=768,
        depth=6
    )
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['custom_vit'])
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize(img_size, interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    
    # Create visualizer
    visualizer = AttentionMapVisualizer(model, patch_size=patch_size, img_size=img_size)
    
    # Visualize attention for each layer and head
    for layer_idx in range(len(model.blocks)):
        # Average of all heads
        fig = visualizer.visualize_average_attention(
            image_path, 
            transform=transform,
            layer_idx=layer_idx,
            save_path=f"attention_avg_layer_{layer_idx}.png"
        )
        plt.close(fig)
        
        # For specific heads
        num_heads = model.num_heads
        for head_idx in range(num_heads):
            fig = visualizer.visualize_average_attention(
                image_path, 
                transform=transform,
                layer_idx=layer_idx,
                head_idx=head_idx,
                save_path=f"attention_layer_{layer_idx}_head_{head_idx}.png"
            )
            plt.close(fig)
    
    # Remove hooks when done
    visualizer.remove_hooks()


# Example of using the visualize_attention_maps function
"""
import timm

# Set paths
model_path = 'path/to/checkpoint.pth.tar'
image_path = 'path/to/image.jpg'

# Get data config
dino_model = timm.create_model('vit_base_patch14_reg4_dinov2.lvd142m', pretrained=False)
data_config = timm.data.resolve_model_data_config(dino_model)

# Visualize attention maps
visualize_attention_maps(model_path, image_path, data_config, patch_size=64, img_size=(2048, 2048))
"""