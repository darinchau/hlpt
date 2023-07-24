from __future__ import annotations
import torch
from typing import Self
from torch import nn
from abc import ABC, abstractmethod as virtual
from torch import Tensor
import numpy as np
import warnings

def _class_name(x) -> str:
    """Returns the model name according to the class name."""
    s = x.__class__.__name__
    # Convert the class name from camel case to words
    w = []
    isup = False
    k = 0
    for i, c in enumerate(s[1:]):
        if isup and c.islower():
            isup = False
            w.append(s[k:i])
            k = i
        if c.isupper():
            isup = True
    w.append(s[k:])
    return ' '.join(w)

def _trainable_cnt(model: nn.Module):
    if isinstance(model, Model):
        return model._num_trainable()
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _nontrain_cnt(model: nn.Module):
    if isinstance(model, Model):
        return model._num_nontrainable()
    else:
        return sum(p.numel() for p in model.parameters() if not p.requires_grad)

def _model_info(m: nn.Module, layers: int) -> str:
    if isinstance(m, Model):
        return m._get_model_info(layers)
    
    assert layers > 0, f"How did you get here?"
    return "  " * (layers - 1) + f"- {_class_name(m)} (Trainable: {_trainable_cnt(m)}, Other: {_nontrain_cnt(m)})"

class Model(nn.Module):
    """Abstract class for all models/model layers etc"""
    # If true, then recursively show the model children details in summary
    _info_show_impl_details = True
    _skip_initialization = False

    def __new__(cls, *args, **kwargs):
        self = super(Model, cls).__new__(cls)
        super().__init__(self)
        if self.forward.__name__ == "_forward_unimplemented":
            raise NotImplementedError(f"The model {_class_name(self)} did not implement the forward method")
        return self
    
    def __init__(self, *args, **kwargs):
        """Creates a new model that can be combined with other models to form bigger models. :)"""
        pass
    
    def _num_trainable(self) -> int:
        total = 0
        for model in self.children():
            total += _trainable_cnt(model)
        return total
    
    def _num_nontrainable(self) -> int:
        total = 0
        for model in self.children():
            total += _nontrain_cnt(model)
        return total
    
    # Recursively get all the model info
    def _get_model_info(self, layers: int):
        if layers == 0:
            s = f"{_class_name(self)} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        else:
            s = "  " * (layers - 1) + f"- {_class_name(self)} (Trainable: {self._num_trainable()}, Other: {self._num_nontrainable()})"
        if self._info_show_impl_details:
            for model in self.children():
                s += "\n"
                s += _model_info(model, layers + 1)
        return s
    
    def summary(self):
        line =  "=" * 100
        return f"""{_class_name(self)}\nModel summary:
{line}
{self._get_model_info(0)}
{line}
"""

    def __repr__(self):
        return _class_name(self)
    
    # The following exists purely to add type annotations
    def __call__(self, *x: Tensor, **kwargs) -> Tensor:
        return super().__call__(*x, **kwargs)

    def to(self, *args, **kwargs) -> Self:
        return super().to(*args, **kwargs)
