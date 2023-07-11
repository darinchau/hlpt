## Contains function related to preprocessing, feature extraction, and data augmentation
## We took the liberty and made everything a function instead of class methods, and rewrote the function from scratch to make them interop with pytorch

import torch
from torch import nn, Tensor
import torchvision.transforms as vision
from hlpt.augments.base import AugmentationLayer
import warnings
from torchvision.transforms._functional_tensor import _get_gaussian_kernel2d

__all__ = (
    "AddCenterCrop", 
    "AddGaussianBlur", 
    "AddRandomChanneling", 
    "AddRandomColoring", 
    "AddRandomCrop", 
    "AddSaltAndPepper",
)

class AddSaltAndPepper(AugmentationLayer):
    """Adds salt and pepper to image/video with some percentage"""
    def __init__(self, p = 0.5, salt_percentage=0.008, pepper_percentage=0.008):
        super().__init__(p)
        assert 0 <= salt_percentage <= 1
        assert 0 <= pepper_percentage <= 1
        self.sp = salt_percentage
        self.pp = pepper_percentage

    def forward(self, x):
        salt_mask = torch.rand_like(x[:, :, :1, :, :]) < self.sp
        pepper_mask = torch.rand_like(x[:, :, :1, :, :]) < self.pp
        salt_mask = torch.cat([salt_mask] * x.shape[2], dim = 2)
        pepper_mask = torch.cat([pepper_mask] * x.shape[2], dim = 2)
        x[salt_mask] = 1
        x[pepper_mask] = 0
        return x

class AddGaussianBlur(AugmentationLayer):
    """Adds Gaussian Blur to the video which should have input dimension (N, frame, C, H, W) ranging from 0 to 1"""
    def __init__(self, p = 0.2, kernel_size = 3, sigma = [0.1, 2.0]):
        super().__init__(p) 
        self.ks = kernel_size
        self.sigma = sigma
        self.pad = nn.ReflectionPad2d([self.ks // 2] * 4)
    
    def forward(self, x: Tensor):
        # Reference from torchvision.transform._functional_tensor because this now support video input
        lx = len(x)
        s = self.rand(self.sigma[0], self.sigma[1])
        kernel = _get_gaussian_kernel2d((self.ks, self.ks), (s, s), dtype=x.dtype, device=x.device)
        kernel = kernel.expand(3, 1, kernel.shape[0], kernel.shape[1])

        x = x.flatten(0, 1)
        x = self.pad(x)
        x = nn.functional.conv2d(x, kernel, groups=x.shape[-3])
        x = x.unflatten(0, (lx, -1))
        return x
    
class AddCenterCrop(AugmentationLayer):
    """Center crops the video and resize it back to full size. Video should have input dimension (N, frame, C, H, W). At most we will crop max_crop_pixels pixels from the border"""
    def __init__(self, input_size: int | tuple[int, int], p = 0.2, max_crop_pixels = 20):
        super().__init__(p) 
        self.maxcrop = max_crop_pixels
        self.resize = vision.Resize(input_size, antialias = True)
    
    def forward(self, x):
        ncrop = int(self.rand(0, self.maxcrop))
        x = x[..., ncrop:-ncrop, ncrop:-ncrop]
        x = self.resize(x)
        return x
    
class AddRandomChanneling(AugmentationLayer):
    """Randomly selects one channel or take the grayscale image :). Video should be (N, frame, C, H, W)"""
    def __init__(self, p = 0.4):
        super().__init__(p)
    
    def forward(self, x):
        n = int(self.rand(0, 5))
        if n == 0: # Consider red channel
            x[:] = x[..., 0:1, :, :]
        elif n == 1: # Consider green channel
            x[:] = x[..., 1:2, :, :]
        elif n == 2: # Consider blue channel
            x[:] = x[..., 2:3, :, :]
        elif n == 3: # Consider bnw image
            x[:] = 0.299 * x[..., 0:1, :, :] + 0.587 * x[..., 1:2, :, :]  + 0.114 * x[..., 2:3, :, :]
        elif n == 4: # Random stuff
            a = self.rand()
            b = self.rand()
            c = a * (1-b)
            d = b * (1-a)
            e = 1-c-d
            x[:] = c * x[..., 0:1, :, :] + d * x[..., 1:2, :, :]  + e * x[..., 2:3, :, :]
        return x
    
class AddRandomCrop(AugmentationLayer):
    """Randomly crop the video and resize it back to full size. Video should have input dimension (N, frame, C, H, W). At most we will crop max_crop_pixels pixels from the border"""
    def __init__(self, input_size: int | tuple[int, int], min_croped_size = 0.6, p = 0.4):
        super().__init__(p)
        assert min_croped_size <= 1
        self.min_croped_size = min_croped_size
        self.resize = vision.Resize(input_size, antialias = True)
        self.input_size = input_size
    
    def forward(self, x: Tensor):
        assert x.shape[-2] == self.input_size[0]
        assert x.shape[-1] == self.input_size[1]

        crop_size = self.rand(self.min_croped_size, 1)
        crop_h = int(x.shape[-2] * crop_size)
        crop_w = int(x.shape[-1] * crop_size)
        crop_start_h = int(self.rand(0, x.shape[-2] - crop_h))
        crop_start_w = int(self.rand(0, x.shape[-1] - crop_w))
        x = x[..., crop_start_h:crop_start_h+crop_h, crop_start_w:crop_start_w+crop_w]
        
        lx = len(x)
        x = x.flatten(start_dim = 0, end_dim = 1)
        x = self.resize(x)
        x = x.unflatten(0, (lx, -1))
        return x

class AddRandomColoring(AugmentationLayer):
    def __init__(self, p = 0.3, brightness = 0.2, contrast = 0.2, hue = 0.2, saturation = 0.2):
        super().__init__(p)
        self.jitter = vision.ColorJitter(brightness, contrast, saturation, hue)
    
    def forward(self, x):
        x = self.jitter(x)
        return x
