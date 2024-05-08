import torch


def heaviside_theta(x, mu, r):
    """Heaviside theta function with parameters mu and r.

    Args:
        x (torch.Tensor): Input tensor.
        mu (float): Center of the function.
        r (float): Width of the function.
    
    Returns:
        torch.Tensor: Output tensor.
    """
    x = x - mu
    return (torch.clamp(x + r, 0, r) - torch.clamp(x, 0, r)) / r

def _linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Note: This function is used to apply the linear interpolation to one element of the input tensor.
    For vectorized operations, use the linear_interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    mu = X
    r = X[1] - X[0]
    F = torch.vmap(heaviside_theta, in_dims=(None, 0, None))
    y = F(x, mu, r).reshape(-1) * Y
    return y.sum()

def linear_interpolation(x, X, Y):
    """Linear interpolation function.

    Args:
        x (torch.Tensor): Input tensor.
        X (torch.Tensor): X values.
        Y (torch.Tensor): Y values.

    Returns:
        torch.Tensor: Output tensor.
    """
    shape = x.shape
    x = x.reshape(-1)
    return torch.vmap(_linear_interpolation, in_dims=(-1, None, None), out_dims=-1)(x, X, Y).reshape(shape)