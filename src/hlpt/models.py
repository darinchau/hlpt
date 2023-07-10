# Commonly used/helpful blocks not included in Pytorch

from hlpt.base import Model, _model_info
from torch import nn, Tensor
import torch
import traceback

class StochasticNode(Model):
    """Stochastic node for VAE. The output is inherently random. Specify the device if needed"""
    _info_show_impl_details = False

    def __init__(self, in_size, out_size):
        # Use tanh for mu to constrain it around 0
        self.lmu = nn.Linear(in_size, out_size)
        self.smu = nn.Tanh()

        # Use sigmoid for sigma to constrain it to positive values and around 1
        self.lsi = nn.Linear(in_size, out_size)
        self.ssi = nn.Sigmoid()

        # Move device to cuda if possible
        self.register_buffer("kl_divergence", torch.tensor(0))

        self.eval_ = False

    def forward(self, x: Tensor):
        mean = self.lmu(x)
        mean = self.smu(mean)

        # sigma to make sigma positive
        var = self.lsi(x)
        var = 2 * self.ssi(var)

        # In evaluation mode, return the mean directly
        if not self.training:
            return mean

        # z = mu + sigma * N(0, 1)
        # if self._N.device != x.device:
        #     self._N = self._N.to(x.device)
        z = mean + var * torch.randn_like(mean)

        # KL divergence
        # https://stats.stackexchange.com/questions/318184/kl-loss-with-a-unit-gaussian?noredirect=1&lq=1
        # https://stats.stackexchange.com/questions/335197/why-kl-divergence-is-non-negative
        # https://kvfrans.com/variational-autoencoders-explained/
        # https://stats.stackexchange.com/questions/318748/deriving-the-kl-divergence-loss-for-vaes/370048#370048
        self.kl_divergence = -.5 * (torch.log(var) - var - mean * mean + 1).sum()

        return z

class Sequential(Model):
    def __init__(self, *f):
        """Sequential model. This has to be overloaded for better error messages and better support for children"""
        for model in f:
            assert isinstance(model, nn.Module)
        self.fs = f

    def forward(self, x):
        for model in self.fs:
            try:
                x = model(x)
            except Exception as e:
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                print(f"Encountered error while running {model}")
                print(f"Traceback: {traceback_str}")
                raise e
        return x
    
    def children(self):
        for i, model in enumerate(self.fs):
            yield model

    def _get_model_info(self, layers: int):
        return "\n".join([_model_info(model, layers) for model in self.fs])

# Modified from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(Model):
    _info_show_impl_details = False
    def __init__(self, dim_model, dropout_p, max_len):
        """Positional encoding useful for transformers"""
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-9.21034037) / dim_model) # That hard coded thing -9.21 is -log(10000)
        
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)
        
    def forward(self, token_embedding: Tensor) -> Tensor:
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])

class LSTMSequenceModel(Model):
    def __init__(self, in_features: int, out_features: int, max_batch_size: int = 32):
        """Takes in vectors of shape (N, in_features), and returns sequence of vectors (N, seq_len, out_features). This can be used as inverse to LSTM"""
        self.rnn = nn.LSTMCell(in_features, out_features)
        self.out_feats = out_features
        # Make the states parameters so the Model.to() method moves it around
        self.register_buffer("hx", torch.zeros(max_batch_size, self.out_feats))
        self.register_buffer("cx", torch.zeros(max_batch_size, self.out_feats))
        self.max_batch_size = max_batch_size

    def forward(self, x: Tensor, sequence_length: int):
        N = len(x)
        if N > self.max_batch_size:
            raise RuntimeError(f"The batch size along the first dimension {N} is greater than the max specified batch size {self.max_batch_size}")
        # Reset state
        self.hx[:] = 0
        self.cx[:] = 0
        
        hx = self.hx[:N]
        cx = self.cx[:N]
        output = []
        for _ in range(sequence_length):
            hx, cx = self.rnn(x, (hx, cx))
            output.append(hx)
        output = torch.stack(output, dim = 1)
        return output

class Ligma(Model):
    """This stands for Linear sigmoid activation, which is defined as:
    
    Ligma(x) = 0 if x < -1, (x+1)/2 if -1 < x < 1, 1 if x > 1
    
    Advantages: Both outputs and gradients are bounded to prevent vanishing gradient
    when x < -1, l(x) = 0 so for sparse entry
    
    Disadvantages: input signal might easily get lost since everything is clamped"""
    def forward(self, x: Tensor):
        x = (torch.abs(x + 1) - torch.abs(x - 1) + 2) / 4
        return x
