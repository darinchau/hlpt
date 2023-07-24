# Commonly used/helpful blocks not included in Pytorch

from hlpt.base import Model, _model_info
from torch import nn, Tensor
import torch
import traceback
import warnings
from itertools import islice
from collections import OrderedDict

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

class Sequential(nn.Sequential, Model):
    _info_show_impl_details = True
    def __init__(self, *f):
        """Sequential model. This has to be overloaded for better error messages"""
        for model in f:
            assert isinstance(model, nn.Module)
        super().__init__(*f)

    def forward(self, x):
        for model in self:
            try:
                x = model(x)
            except Exception as e:
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                print(f"Encountered error while running {model}")
                print(f"Traceback: {traceback_str}")
                raise e
        return x

    def _get_model_info(self, layers: int):
        return "\n".join([_model_info(model, layers) for model in self])
    
class ModelList(nn.ModuleList, Model):
    def _get_model_info(self, layers: int):
        return "\n".join([_model_info(model, layers) for model in self])
    
    def forward(self, x: Tensor):
        for model in self:
            try:
                x = model(x)
            except Exception as e:
                traceback_str = ''.join(traceback.format_tb(e.__traceback__))
                print(f"Encountered error while running {model}")
                print(f"Traceback: {traceback_str}")
                raise e
        return x

class ExtractPrincipalComponent(Model):
    """Takes the tensor x, and returns the principal d dimensions by calculating its covariance matrix. d indicates the number of principal components to extract
    Input shape should be (N, nfeatures)"""
    def __init__(self):
        self.eigenvectors = None

    def fit(self, x: Tensor):
        """X is a (..., n_features). This computes the projection data for PCA. Returns None. The sorted eigenvalues is at `model.eigenvalues` and the sorted eigenvectors are at `model.eigenvectors`"""
        # Flatten all dimensions first
        xs = x.shape
        x = x.flatten(0, -2)
        cov = torch.cov(x.T.float())
        l, v = torch.linalg.eig(cov)

        # Temporarily supress warnings. This normally screams at us for discarding the complex part. But covariance matric is always positive definite so real eigenvalues :)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.eigenvalues, sorted_eigenidx = torch.abs(l.double()).sort(descending=True)
            self.eigenvectors = v[:, sorted_eigenidx].double()
        
        x = x.unflatten(0, xs[:-1])
        
    def project(self, x: Tensor) -> Tensor:
        """X is a (..., n_features) tensor. This performs the projection for you and returns an (n_data, d) tensor. Raises a runtime error if n_features does not match that in training"""
        if self.eigenvectors is None:
            raise RuntimeError("Projection data has not been calculated yet. Please first call model.fit()")
        
        P = self.eigenvectors
        
        if x.shape[1] != P.shape[0]:
            raise RuntimeError(f"Expects {P.shape[0]}-dimensional data due to training. Got {x.shape[1]}-d data instead.")
        
        # Welcome to transpose hell
        X_ = x - torch.mean(x.float(), dim = 0)
        components = X_.double() @ P
        components = components / torch.max(torch.abs(components))
        return components
    
    def unproject(self, x: Tensor):
        """Try to compute the inverse of model.project(X). The input is a tensor of shape (n_data, d) and returns a tensor of (n_data, n_features)"""
        # XP = X* so given X* we have X = X*P⁻¹
        # Problem is P is a matrix of shape (n_features, d), so we need to make it square first to take inverse.
        # However, P is originally (n_features, n_features) big which we can take inverses, the reason
        # P has the shape (n, d) is because it is actually the combination of the real P matrix followed by extracting first N columns
        # We use a workaround: append zeros on X until it has enough features, then use the full P inverse
        if self.eigenvectors is None:
            raise RuntimeError("Projection data has not been calculated yet. Please first call model.fit()")

        X_ = torch.zeros(x.shape[0], self.eigenvalues.shape[0])
        X_[:, :x.shape[1]] = x
        P = self.eigenvectors
        try:
            result = X_ @ torch.linalg.inv(P)
        except RuntimeError as e:
            raise RuntimeError("PCA eigenvectors matrix is not invertible for some reason. This is probably due to that there are very very very small (coerced to 0) eigenvalues.")
        return result
        
    def forward(self, x: Tensor) -> Tensor:
        """fit followed by project. Takes input of shape (N, nfeatures)"""
        self.fit(x)
        return self.project(x)

class Normalize(Model):
    def __init__(self, min_ = 0, max_ = 1):
        self.min = min_
        self.max = max_
    
    def forward(self, x: Tensor) -> Tensor:
        return (x - x.min()) / (x.max() - x.min()) * (self.max - self.min) + self.min
