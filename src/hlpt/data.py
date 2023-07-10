### Provides a custom Dataset class for nice things to happen
from __future__ import annotations
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import Tensor
from sklearn.model_selection import train_test_split
import numpy as np
from tqdm import tqdm as ProgressBar

class DataIterator(DataLoader):
    """A wrapper for datasets and dataloaders that automatically perform train-test splits and stuff like a torch dataloader
    
    label: if not None, we will perform train-test split in a stratified manner"""
    def __init__(self, *t : Tensor, batch_size: int = 1, label: list | None = None, shuffle = False, progress_bar = True, use_multiprocess = True):
        assert len(t) > 0
        self.len_t = len(t[0])
        for tensor in t:
            assert len(tensor) == self.len_t
        
        super().__init__(TensorDataset(*t), batch_size=batch_size, shuffle = shuffle, num_workers=1 if use_multiprocess else 0)
        self.label = label
        self.pbar = progress_bar
        self.shuffle = shuffle
        self.use_multiprocess = use_multiprocess
        self.bs = batch_size

    def split(self, train_size = 0.85, *, seed = None) -> tuple[DataIterator, DataIterator]:
        """Performs train-test split"""
        trains = []
        tests = []

        if seed is None:
            seed = 42069

        for tensor in self.tensors:
            train, test = train_test_split(tensor.detach().cpu().numpy(), test_size = 1 - train_size, random_state=seed, shuffle = False, stratify=self.label)
            train = torch.as_tensor(train, dtype = tensor.dtype, device = tensor.device)
            test = torch.as_tensor(test, dtype = tensor.dtype, device = tensor.device)
            trains.append(train)
            tests.append(test)
        
        # Create new TensorDatasets and DataLoaders for train and test sets
        train_loader = DataIterator(*trains, batch_size=self.bs, label=self.label, shuffle=self.shuffle, progress_bar=self.pbar, use_multiprocess=self.use_multiprocess)
        test_loader = DataIterator(*tests, batch_size=self.bs, label=self.label, shuffle=self.shuffle, progress_bar=self.pbar, use_multiprocess=self.use_multiprocess)
        return train_loader, test_loader
    
    @property
    def tensors(self):
        return self.dataset[:]
    
    def __iter__(self):
        if not self.pbar:
            return super().__iter__()
        
        p = ProgressBar(total = len(self))
        for x in super().__iter__():
            yield x
            if isinstance(x, list):
                p.update(x[0].size(0))
            elif isinstance(x, Tensor):
                p.update(x.size(0))
        p.close()
