from hlpt.base import Model, _model_info
from torch import nn, Tensor
import torch
import traceback

def random_(min_ = 0, max_ = 1) -> float:
    assert min_ < max_
    if min_ == 0 and max_ == 1:
        return torch.rand(1).item()
    return torch.rand(1).item() * (max_ - min_) + min_

def is_fucked(t):
    return torch.isnan(t).any() or torch.isinf(t).any()

class AugmentationLayer(Model):
    """Represents an Augmentation layer which randomly applys stuff"""
    def __init__(self, p: float):
        """Initialize the augmentation layer with a 0 <= p < 1 indicating how likely is it we apply this augmentation"""
        assert 0 <= p <= 1
        self.p = p

    # This exists purely to overload the signature
    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
        
    def rand(self, min_ = 0, max_ = 1) -> float:
        """Get a random number from torch. This exists so that we always get the same random number from the same rng"""
        return random_(min_ = min_, max_ = max_)
    
class Preprocessor(Model):
    _info_show_impl_details = False
    """At most apply a random subset of n models"""
    def __init__(self, *models: AugmentationLayer, at_most_apply: int = 3, allow_nan: bool = False):
        for model in models:
            assert isinstance(model, AugmentationLayer)
        self.models = models
        self.at_most_apply = at_most_apply
        self.traces = []
        self.allow_nan = allow_nan
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        # Applies a random subset of models to the input tensor during training.
        applied = []
        considered = []
        while len(considered) < len(self.models) and len(applied) < self.at_most_apply:
            index = int(random_(0, len(self.models)))
            if index in considered:
                continue
            considered.append(index)
            if random_() < self.models[index].p:
                applied.append(index)
                with torch.inference_mode():
                    x = self.models[index](x)
                    if not self.allow_nan and is_fucked(x):
                        raise RuntimeError(f"""We found a NaN or Infinity value in the tensor after applying these augmentations... Don't worry, its our fault, or yours if you wrote your own augmentation
                                           Set allow_nan=True if you want to suppress this runtime error
                                           Trace:
                                           {[self.models[a].__repr__() for a in applied]}""")
        return x

    def children(self):
        for i, model in enumerate(self.models):
            yield model
