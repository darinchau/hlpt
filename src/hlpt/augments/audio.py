## Contains function related to preprocessing, feature extraction, and data augmentation
## We took the liberty and made everything a function instead of class methods, and rewrote the function from scratch to make them interop with pytorch

import torch
from torch import nn, Tensor
import torchaudio
from torchaudio.transforms import *
from hlpt.base import Model
import warnings
from hlpt.augments.base import AugmentationLayer
import torchaudio.functional as F

__all__ = (
    "AddNoise", 
    "AddPitchScaling", 
    "AddRandomGain", 
    "AddWhiteNoise",
    "AddReverb",
    "AddLowPassFilter",
    # "AddHighPassFilter", # Found bug, fix in future
    "AddWhiteNoisePadding",
    "AudioAugmentationLayer"
)

class AudioAugmentationLayer(AugmentationLayer):
    def __call__(self, x: Tensor):
        """Augmentation on raw audio data. Accepts audio of shape (..., nchannels, nframes)"""
        if len(x.shape) < 3:
            raise RuntimeError(f"{self.__repr__()} accepts audio of shape (..., nchannels, nbins), found x with shape {x.shape}")
        s = x.shape[:-2]
        x = x.flatten(0, -3)
        y = super().__call__(x)
        y = y.unflatten(0, s)
        return y

class AddPitchScaling(AudioAugmentationLayer):
    def __init__(self, sample_rate_hz: int = 22050, p = 0.1):
        """Pytorch augmentation layer that applies pitch shifting. This is hardcoded to be between -2 halfsteps and 2 halfsteps"""
        super().__init__(p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.shift_dflat = PitchShift(sample_rate=sample_rate_hz, n_steps = -2)
            self.shift_flat = PitchShift(sample_rate=sample_rate_hz, n_steps = -1)
            self.shift_sharp = PitchShift(sample_rate=sample_rate_hz, n_steps = 1)
            self.shift_dsharp = PitchShift(sample_rate=sample_rate_hz, n_steps = 2)

    def forward(self, x: Tensor):
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        rng = int(self.rand(0, 4))
        if rng == 0:
            x = self.shift_dflat(x)
        elif rng == 1:
            x = self.shift_flat(x)
        elif rng == 2:
            x = self.shift_sharp(x)
        else:
            x = self.shift_dsharp(x)
        x = x.unflatten(dim = 0, sizes = xs)
        return x

class AddWhiteNoise(AudioAugmentationLayer):
    def __init__(self, min_amplitude = 0.001, max_amplitude = 0.015, p = 0.5) -> None:
        """PyTorch augmentation layer that adds Gaussian noise to an audio waveform."""
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def forward(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x)
        amplitude = self.rand(self.min_amplitude, self.max_amplitude)
        x = x + amplitude * noise
        return x
    
class AddRandomGain(AudioAugmentationLayer):
    def __init__(self, p = 0.6, min_ = 0.8, max_ = 1.2):
        """PyTorch augmentation layer that applies random gain to an audio waveform."""
        super().__init__(p)
        self.min_gain_factor = min_
        self.max_gain_factor = max_
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.rand(self.min_gain_factor, self.max_gain_factor)
        return x

class AddLowPassFilter(AudioAugmentationLayer):
    def __init__(self, sample_rate_hz: int, p = 0.4):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
    
    def forward(self, x: Tensor):
        filter = self.rand(300, 4000)
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        x = F.lowpass_biquad(x, self.sample_rate_hz, filter)
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x

class AddReverb(AudioAugmentationLayer):
    def __init__(self, sample_rate_hz: int, p = 0.4, max_delay_ms = 60, max_reverb = 0.1):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
        assert max_delay_ms >= 0
        self.max_delay_ms = max_delay_ms
        self.max_reverb = max_reverb
    
    def forward(self, x: Tensor):
        delay_frames = max(int(self.rand() * self.sample_rate_hz / 1000 * self.max_delay_ms), 1)
        reverb_amount = self.rand(-self.max_reverb, self.max_reverb)
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        x[..., delay_frames:] += x[..., :-delay_frames] * reverb_amount
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x
    
class AddWhiteNoisePadding(AudioAugmentationLayer):
    def __init__(self, sample_rate_hz: int, max_second: float = 0.1, min_amplitude = 0.1, max_amplitude = 0.2, p = 0.3):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
        self.max_second = max_second
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def forward(self, x: Tensor):
        length = int(self.sample_rate_hz * self.max_second * self.rand())
        x = nn.functional.pad(x, (length//2, length//2), value = 0)
        amplitude = self.rand(self.min_amplitude, self.max_amplitude)
        noise = torch.rand_like(x)
        x = x + amplitude * noise
        return x
