Some high-level commonly used pytorch components with model.summary() now a thing.

```python
from hlpt import Model
import torch

class MyModel(Model):
    # Model is a child class of nn.Module
    def __init__(self, ...):
        # Now we do not need to call super().__init__() because this is called for you
        # But something something MRO I know right so Model being the direct child of nn.Module should call the right thing
        # As long as you dont use diamond inheritance with Model being one of your branch
        ...
    
    def forward(self, ...):
        # Define forward in the usual way
        ...
    

x = torch.randn((3, 5))

# Model's operator() methods now have type hints about having tensors as output
y = model(x)

# Model.summary returns a string akin to the Keras summary() method
print(model.summary())
```