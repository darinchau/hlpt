
import torch
from torch import nn, Tensor
import torchaudio
from torchaudio.transforms import *
from hlpt.base import Model
import warnings
from hlpt.augments.base import AugmentationLayer
import torchaudio.functional as F

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

class AddFFTTimeStretch(AugmentationLayer):
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

class AddFFTFrequencyMasking(AugmentationLayer):
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

class AddFFTTimeMasking(AugmentationLayer):
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
