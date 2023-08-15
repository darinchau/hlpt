from __future__ import annotations
import matplotlib.pyplot as plt
from torch import Tensor
import torch
import gc

class History:
    """A helper class to plot the history of training"""
    def __init__(self):
        self.losses: list[dict[str, float]] = [{}]
        self.counts = {}
        self.names: set[str] = set()
        self.logs: list[str] = []
        self.training = True
        self._test_called = False
    
    def train(self):
        self.training = True
        if self._test_called:
            self._test_called = False
            self.update()
        
    def test(self):
        self.training = False
        self._test_called = True

    def add_loss(self, name, loss, batch_size = 1):
        """Logs the loss to an internal buffer for plotting later"""
        name_ = f"Train {name}" if self.training else f"Test {name}"
        self.names.add(name_)

        if isinstance(loss, Tensor):
            loss = loss.item()
        
        if name_ in self.counts:
            n = self.counts[name_]
            self.losses[-1][name_] = (n * self.losses[-1][name_] + loss) / (n + batch_size)
            self.counts[name_] += batch_size
        else:
            self.losses[-1][name_] = loss
            self.counts[name_] = batch_size

    def update(self):
        """Call this at the start of every epoch"""
        self.losses.append({})
        self.counts = {}

    def plot(self, root: str, n: str):
        """Plot the losses using matplotlib"""
        fig, ax = plt.subplots()
        for name in self.names:
            x, y = [], []
            for i, losses in enumerate(self.losses):
                if name in losses:
                    x.append(i)
                    y.append(losses[name])
            ax.plot(x, y, label = name)

        ax.set_yscale('log')
        ax.legend()
        ax.set_title(f"Train/Test Error plot")
        fig.savefig(f"{root}/{n} training loss.jpg")
        plt.close('all')

    def __iter__(self):
        for i, losses in enumerate(self.losses):
            s = f"On epoch {i}: "
            s += ", ".join([f"{k} = {v:.6f}"for k, v in losses.items()])
            yield s
        
        yield "\n"
        for s in self.logs:
            yield s

    def current_epoch(self, current_epoch: int = -1) -> str:
        """Print the current epoch loss information. If current epoch < 0, then we help you calculate the epoch. Otherwise override the epoch for you, which would be the zero-count epoch number."""
        s = f"Epoch {current_epoch if current_epoch > 0 else len(self.losses) - 1}: ".ljust(13)
        for name, loss in self.losses[-1].items():
            s += f"{name} = {loss:.4f} "
        return s

# A helper class to monitor cuda usage for debugging
# Use this with the debugger to create an ad hoc cuda memory watcher in profile txt
class CudaMonitor:
    # Property flag forces things to save everytime a line of code gets run in the debugger
    @property
    def memory(self):
        print("Logging memory")
        s = []
        num_tensors = 0
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    # Total numbers
                    total = 1
                    for i in obj.shape:
                        total *= i
                    s.append((total, f"Tensor: {type(obj)}, size: {obj.size()}, shape: {obj.shape}"))
                    num_tensors += 1
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                pass
        s = [x[1] for x in sorted(s, key = lambda a: a[0], reverse = True)]
        with open("profile.txt", 'w') as f:
            f.write(f"Memory allocated: {torch.cuda.memory_allocated()}\n")
            f.write(f"Max memory allocated: {torch.cuda.max_memory_allocated()}\n")
            for y in s:
                f.write(y)
                f.write("\n")
        return f"Logged {num_tensors} tensors at profile.txt"
    
    def clear(self):
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    del obj
            except KeyboardInterrupt:
                raise KeyboardInterrupt
            except Exception as e:
                pass
