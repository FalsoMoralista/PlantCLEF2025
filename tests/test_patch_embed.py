import torch
import sys
sys.path.append('../')

from src.models.patch_embed import PatchEmbed


embed = PatchEmbed(img_size=(2048,1024), patch_size=16, in_chans=3)
batch_size = 1
x = torch.randn(batch_size, 3, 2048, 1024)  # Batch of 64 samples
print(embed(x).size())