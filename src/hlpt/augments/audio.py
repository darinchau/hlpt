## Contains function related to preprocessing, feature extraction, and data augmentation
## We took the liberty and made everything a function instead of class methods, and rewrote the function from scratch to make them interop with pytorch

import torch
from torch import nn, Tensor
import torchaudio
from torchaudio.transforms import *
from hlpt import Model
import warnings
from hlpt import AugmentationLayer

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
        """PyTorch augmentation layer that applies time stretching. Note this"""
        super().__init__(p)
        self.t = TimeStretch(hop_length = 256, n_freq = nbins)
        self.min_ = min_
        self.max_ = max_

    def forward(self, x: Tensor):
        if len(x.shape) != 3:
            raise RuntimeError("Number of dimensions of x must be 3 (N, n_audioframes, n_melfilterbanks)")

        rate = self.rand(self.min_, self.max_)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x = self.t(x, rate).to(dtype = x.dtype)
        return x

class AddPitchScaling(AugmentationLayer):
    def __init__(self, sample_rate: int = 22050, p = 0.1):
        """Pytorch augmentation layer that applies pitch shifting. This is hardcoded to be between -2 halfsteps and 2 halfsteps"""
        super().__init__(p)
        # Temporarily shut up pls
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.shift_dflat = PitchShift(sample_rate=sample_rate, n_steps = -2)
            self.shift_flat = PitchShift(sample_rate=sample_rate, n_steps = -1)
            self.shift_sharp = PitchShift(sample_rate=sample_rate, n_steps = 1)
            self.shift_dsharp = PitchShift(sample_rate=sample_rate, n_steps = 2)

    def forward(self, x: Tensor):
        if len(x.shape) != 2:
            raise RuntimeError("Number of dimensions of x must be 2 (N, n_audioframes)")
        rng = int(self.rand(0, 4))
        if rng == 0:
            x = self.shift_dflat(x)
        elif rng == 1:
            x = self.shift_flat(x)
        elif rng == 2:
            x = self.shift_sharp(x)
        else:
            x = self.shift_dsharp(x)
        return x

class AddWhiteNoise(AugmentationLayer):
    def __init__(self, min_amplitude = 0.001, max_amplitude = 0.015, p = 0.5) -> None:
        """Pytorch augmentation layer that adds white noise. 
        min_amplitude: The minimum amplitude of the added white noise. This parameter controls the
        intensity of the noise added to the audio file
        max_amplitude: The maximum amplitude of the added white noise. This parameter controls the
        strength of the noise added to the audio file
        p_apply: the probability of applying this transform"""
        super().__init__()
        assert 0 <= p <= 1
        self.p = p
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def forward(self, x: Tensor) -> Tensor:
        noise = torch.randn_like(x)
        amplitude = self.rand(self.min_amplitude, self.max_amplitude)
        x = x + amplitude * noise
        return x
    
class AddRandomGain(AugmentationLayer):
    """PyTorch augmentation layer that applies random gain and adds Gaussian noise to an audio waveform."""
    def __init__(self, p = 0.6, min_ = 0.8, max_ = 1.2):
        super().__init__()
        assert 0 <= p <= 1
        self.p = p
        self.min_gain_factor = min_
        self.max_gain_factor = max_
    
    def forward(self, x: Tensor) -> Tensor:
        x = x * self.rand(self.min_gain_factor, self.max_gain_factor)
        return x
