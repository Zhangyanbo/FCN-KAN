import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(x, w1, w2, b1, b2, n_sin):
    """
    phi function that integrates sinusoidal embeddings with MLP layers.

    Args:
        x (torch.Tensor): Input tensor.
        w1 (torch.Tensor): Weight matrix for the first linear transformation.
        w2 (torch.Tensor): Weight matrix for the second linear transformation.
        b1 (torch.Tensor): Bias vector for the first linear transformation.
        b2 (torch.Tensor): Bias vector for the second linear transformation.
        n_sin (int): Number of sinusoidal functions to generate.

    Returns:
        torch.Tensor: Transformed tensor.
    """
    omega = (2 ** torch.arange(0, n_sin)).float().reshape(-1, 1)
    omega_x = F.linear(x, omega, bias=None)
    x = torch.cat([x, torch.sin(omega_x), torch.cos(omega_x)], dim=-1)
    
    x = F.linear(x, w1, bias=b1)
    x = F.silu(x)
    x = F.linear(x, w2, bias=b2)
    return x


class KANLayer(nn.Module):
    """
    A layer in a Kolmogorov–Arnold Networks (KAN).

    Attributes:
        dim_in (int): Dimensionality of the input.
        dim_out (int): Dimensionality of the output.
        fcn_hidden (int): Number of hidden units in the feature transformation.
        fcn_n_sin (torch.tensor): Number of sinusoidal functions to be used in phi.
    """
    def __init__(self, dim_in, dim_out, fcn_hidden=32, fcn_n_sin=3):
        """
        Initializes the KANLayer with specified dimensions and sinusoidal function count.
        
        Args:
            dim_in (int): Dimension of the input.
            dim_out (int): Dimension of the output.
            fcn_hidden (int): Number of hidden neurons in the for the learned non-linear transformation.
            fcn_n_sin (int): Number of sinusoidal embedding frequencies.
        """
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden, 1+fcn_n_sin*2))
        self.W2 = nn.Parameter(torch.randn(dim_in, dim_out, 1, fcn_hidden))
        self.B1 = nn.Parameter(torch.randn(dim_in, dim_out, fcn_hidden))
        self.B2 = nn.Parameter(torch.randn(dim_in, dim_out, 1))

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.fcn_hidden = fcn_hidden
        self.fcn_n_sin = torch.tensor(fcn_n_sin).long()

        self.init_parameters()
    
    def init_parameters(self):
        nn.init.xavier_normal_(self.W1)
        nn.init.xavier_normal_(self.W2)
        # apply zero bias
        nn.init.zeros_(self.B1)
        nn.init.zeros_(self.B2)
    
    def map(self, x):
        """
        Maps input tensor x through phi function in a vectorized manner.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after mapping through phi.
        """
        F = torch.vmap(
            # take dim_in out, -> dim_in x (dim_out, *)(1)
            torch.vmap(phi, (None, 0, 0, 0, 0, None), 0), # take dim_out out, -> dim_out x (*)
            (0, 0, 0, 0, 0, None), 0
            )
        return F(x.unsqueeze(-1), self.W1, self.W2, self.B1, self.B2, self.fcn_n_sin).squeeze(-1)

    def forward(self, x):
        """
        Forward pass of the KANLayer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Summed output after mapping each dimensions through phi.
        """
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        batch, dim_in = x.shape
        assert dim_in == self.dim_in

        batch_f = torch.vmap(self.map, 0, 0)
        phis = batch_f(x) # [batch, dim_in, dim_out]

        return phis.sum(dim=1)
    
    def take_function(self, i, j):
        """
        Returns a phi function specific to the (i, j)-th elements of parameters.

        Args:
            i (int): Row index in parameter tensors.
            j (int): Column index in parameter tensors.

        Returns:
            function: A function that computes phi for specific parameters.
        """
        def activation(x):
            return phi(x, self.W1[i, j], self.W2[i, j], self.B1[i, j], self.B2[i, j], self.fcn_n_sin)
        return activation