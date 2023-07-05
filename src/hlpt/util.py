from __future__ import annotations
import matplotlib.pyplot as plt
from torch import Tensor

class History:
    """A helper class to plot the history of training"""
    def __init__(self):
        self.losses: list[dict[str, float]] = [{}]
        self.counts = {}
        self.names: set[str] = set()
        self.logs: list[str] = []
        self.training = True
    
    def train(self):
        self.training = True

    def test(self):
        self.training = False

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

    def plot(self, root: str, name: str):
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
        fig.savefig(f"{root}/{name} training loss.jpg")

    def __iter__(self):
        for i, losses in enumerate(self.losses):
            s = f"On epoch {i}: "
            s += ", ".join([f"{k} = {v:.6f}"for k, v in losses.items()])
            yield s
        
        yield "\n"
        for s in self.logs:
            yield s

    def current_epoch(self) -> str:
        """Print the current epoch loss information"""
        s = f"Epoch {len(self.losses) - 1}: ".ljust(13)
        for name, loss in self.losses[-1].items():
            # if self.training and name.startswith("Test"):
            #     continue
            # elif not self.training and name.startswith("Train"):
            #     continue
            s += f"{name} = {loss:.4f} "
        return s
