import torch
from torchvision import transforms
from timm.data import create_transform
from einops import rearrange
from PIL import Image
import timm

class Patchify:
    def __init__(self, n):
        self.n = n

    def __call__(self, tensor):
        # tensor: (C, H, W)
        C, H, W = tensor.shape
        assert H % self.n == 0 and W % self.n == 0, "Image dimensions must be divisible by n"
        patch_h, patch_w = H // self.n, W // self.n
        patches = rearrange(tensor, 'c (h ph) (w pw) -> (h w) c ph pw',
                            h=self.n, w=self.n, ph=patch_h, pw=patch_w)
        return patches  # shape: (n*n, C, patch_h, patch_w)
