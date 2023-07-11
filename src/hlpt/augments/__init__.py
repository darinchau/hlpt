## Provides a bunch of different augmentation layers for use

from hlpt.augments.base import Preprocessor, AugmentationLayer
from hlpt.augments.audio import ExtractFFT, AddNoise, AddPitchScaling, AddRandomGain, AddTimeStretch, AddWhiteNoise
from hlpt.augments.video import AddCenterCrop, AddGaussianBlur, AddRandomChanneling, AddRandomColoring, AddRandomCrop, AddSaltAndPepper