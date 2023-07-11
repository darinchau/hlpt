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
            x = self.t(x.transpose(-1, -2), rate).to(dtype = x.dtype).transpose(-1, -2)
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

class AddPitchScaling(AugmentationLayer):
    def __init__(self, sample_rate_hz: int = 22050, p = 0.1):
        """Pytorch augmentation layer that applies pitch shifting. This is hardcoded to be between -2 halfsteps and 2 halfsteps"""
        super().__init__(p)
        # Temporarily shut up pls
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
        filter = self.rand(30, 4000)
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        x = F.lowpass_biquad(x, self.sample_rate_hz, filter)
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x

class AddHighPassFilter(AugmentationLayer):
    """Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
    def __init__(self, sample_rate_hz: int, p = 0.4):
        super().__init__(p)
        self.sample_rate_hz = sample_rate_hz
    
    def forward(self, x: Tensor):
        filter = self.rand(4000, 20000)
        xs = x.shape[:-1]
        x = x.flatten(end_dim=-2)
        x = F.highpass_biquad(x, self.sample_rate_hz, filter)
        x = x.to(x.device, dtype = x.dtype).unflatten(dim = 0, sizes = xs)
        return x

class AddReverb(AugmentationLayer):
    """Audio feature extraction. Accepts audio of shape (N, nchannels, nframes)"""
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