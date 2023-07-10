from hlpt.base import Model, _model_info
from torch import nn, Tensor
import torch
import traceback

def random_(min_ = 0, max_ = 1) -> float:
    assert min_ < max_
    return torch.rand(1).item() * (max_ - min_) + min_

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
    """At most apply n modelss"""
    def __init__(self, *models: AugmentationLayer, at_most_apply: int = 3):
        for model in models:
            assert isinstance(model, AugmentationLayer)
        self.models = models
        self.at_most_apply = at_most_apply
    
    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            return x
        
        applied = set()
        considered = set()
        while len(considered) < len(self.models) and len(applied) < self.at_most_apply:
            index = int(random_(0, len(self.models)))
            if index in considered:
                continue
            considered.add(index)
            if random_() < self.models[index].p:
                if hasattr(self, "debug") and self.debug:
                    print(f"Applying {self.models[index]}")
                applied.add(index)
                x = self.models[index](x)
        return x
