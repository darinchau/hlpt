## Contains function related to preprocessing, feature extraction, and data augmentation
## We took the liberty and made everything a function instead of class methods, and rewrote the function from scratch to make them interop with pytorch

import torch
from torch import nn, Tensor
import torchaudio
from torchaudio.transforms import *
from hlpt.base import Model
import warnings
from hlpt.augments.base import AugmentationLayer

__all__ = (
    "ExtractFFT", 
    "AddNoise", 
    "AddPitchScaling", 
    "AddRandomGain", 
    "AddTimeStretch", 
    "AddWhiteNoise",
    "AddReverb",
    "AddLowPassFilter",
    "AddTimeMasking",
    "AddFrequencyMasking"
)

class ExtractFFT(Model):
    """Performs FFT to extract features in mel scale on the provided audio"""
    def __init__(self, sample_rate_hz = 22050, nbins = 128):
        self.melspec = MelSpectrogram(sample_rate = sample_rate_hz, n_mels = nbins, hop_length=256, n_fft = 1024)
        self.db = AmplitudeToDB(top_db=80)

    def forward(self, x: Tensor):
        x = self.melspec(x)
        x = self.db(torch.abs(x))
        # This exists for normalization
        x -= torch.max(x)
        x[x < -80] = -80
        return torch.transpose(x, 1, 2)

class AddTimeStretch(AugmentationLayer):
    def __init__(self, min_ = 0.8, max_ = 1.2, p = 0.5, nbins = 128):
        """PyTorch augmentation layer that applies time stretching on mel spectrogram. Accepts audio of shape (N, nfrequency, nbins)"""
        super().__init__(p)
        self.t = TimeStretch(hop_length = 256, n_freq = nbins)
        self.min_ = min_
        self.max_ = max_
        self.nbins = nbins

    def forward(self, x: Tensor):
        if len(x.shape) < 3:
            raise RuntimeError(f"x must have at least 3 dimensions, but found x with shape {x.shape}")
        nbins = x.shape[-1]
        if self.nbins != nbins:
            warnings.warn(f"The declared number of bins {self.nbins} does not match x {x.shape[-1]} in AddFrequencyName.")


        rate = self.rand(self.min_, self.max_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.t(x, rate).to(dtype = x.dtype)
        return x

class AddFrequencyMasking(AugmentationLayer):
    def __init__(self, p = 0.4, max_width_percentage = 0.08, nbins = 128):
        """PyTorch augmentation layer that applies frequency masking on mel spectrogram. Accepts audio of shape (N, nfrequency, nbins)"""
        super().__init__(p)
        assert 0 < max_width_percentage < 1
        self.max_ = max_width_percentage
        self.nbins = nbins

    def forward(self, x: Tensor):
        if len(x.shape) < 3:
            raise RuntimeError(f"x must have at least 3 dimensions, but found x with shape {x.shape}")
        nbins = x.shape[-1]
        if self.nbins != nbins:
            warnings.warn(f"The declared number of bins {self.nbins} does not match x {x.shape[-1]} in AddFrequencyName.")
        
        mask = int(nbins * self.max_ * self.rand())
        bottom = int(self.rand() * (nbins - mask))
        x[..., bottom:bottom+mask] = torch.min(x)
        return x
    
class AddTimeMasking(AugmentationLayer):
    def __init__(self, p = 0.4, max_width_percentage = 0.08, nbins = 128):
        """PyTorch augmentation layer that applies time masking on mel spectrogram. Accepts audio of shape (N, nfrequency, nbins)"""
        super().__init__(p)
        assert 0 < max_width_percentage < 1
        self.max_ = max_width_percentage
        self.nbins = nbins

    def forward(self, x: Tensor):
        if len(x.shape) < 3:
            raise RuntimeError(f"x must have at least 3 dimensions, but found x with shape {x.shape}")
        nbins = x.shape[-1]
        if self.nbins != nbins:
            warnings.warn(f"The declared number of bins {self.nbins} does not match x {x.shape[-1]} in AddFrequencyName.")
        
        naudioframes = x.shape[-2]
        mask = int(naudioframes * self.max_ * self.rand())
        left = int(self.rand() * (naudioframes - mask))
        x[..., left:left+mask, :] = torch.min(x)
        return x

# Add 1 and 2d support to pitch scaling stuff
class AddPitchScaling(AugmentationLayer):
    """PyTorch augmentation layer that applies random gain and adds Gaussian noise to an audio waveform.
        Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
    def __init__(self, sample_rate_hz: int, min_cent = -200, max_cent = 200, p = 0.4):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
        self.min_x = min_cent
        self.max_x = max_cent
    
    def forward(self, x: Tensor):
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2).detach().cpu()
        cent = int(self.rand(self.min_x, self.max_x))
        x, _ = torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate_hz, [["pitch", f"{cent}"]])
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x

class AddWhiteNoise(AugmentationLayer):
    def __init__(self, min_amplitude = 0.001, max_amplitude = 0.015, p = 0.5) -> None:
        """Pytorch augmentation layer that adds white noise. 
        min_amplitude: The minimum amplitude of the added white noise. This parameter controls the
        intensity of the noise added to the audio file
        max_amplitude: The maximum amplitude of the added white noise. This parameter controls the
        strength of the noise added to the audio file
        p_apply: the probability of applying this transform
        
        PyTorch augmentation layer that applies random gain and adds Gaussian noise to an audio waveform.
        Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
        super().__init__(p)
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def forward(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x)
        amplitude = self.rand(self.min_amplitude, self.max_amplitude)
        x = x + amplitude * noise
        return x
    
class AddRandomGain(AugmentationLayer):
    """PyTorch augmentation layer that applies random gain and adds Gaussian noise to an audio waveform.
    Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
    def __init__(self, p = 0.6, min_ = 0.8, max_ = 1.2):
        super().__init__(p)
        self.min_gain_factor = min_
        self.max_gain_factor = max_
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.rand(self.min_gain_factor, self.max_gain_factor)
        return x

class AddReverb(AugmentationLayer):
    """Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
    def __init__(self, sample_rate_hz: int, p = 0.4):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
    
    def forward(self, x: Tensor):
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2).detach().cpu()
        x, _ = torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate_hz, [["reverb", "-w"]])
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x

class AddLowPassFilter(AugmentationLayer):
    """Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
    def __init__(self, sample_rate_hz: int, p = 0.4):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
    
    def forward(self, x: Tensor):
        filter = int(self.rand(30, 4000))
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2).detach().cpu()
        x, _ = torchaudio.sox_effects.apply_effects_tensor(x, self.sample_rate_hz, [["lowpass", "-1", f"{filter}"]])
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x
