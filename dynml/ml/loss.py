"""Define all loss function related code.

This module contains all loss function related code.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import mean, sum, Tensor
# import internal python-package code
# None


# export public code
__all__ = ["recon_loss"]


# define loss functions
def recon_loss(Y: Tensor, X: Tensor) -> Tensor:
    """Return the reconstruction loss.

    This method returns the reconstruction loss

    .. math::
        \\mathcal{L}_{\\text{recon}} =
        \\frac{1}{N} \\sum_{i=1}^{N} \\|y_i - x_i\\|_2^2,

    where :math:`N` is ``X.shape[0:-1]`` and ``(x_i, y_i)`` is an input and
    target data pair.

    | **Args**
    |   ``Y`` (``Tensor``): the target data with shape ``(...,) + (dim,)``
    |   ``X`` (``Tensor``): the input data with shape ``(...,) + (dim,)``

    | **Returns**
    |   ``Tensor``: the reconstruction loss

    | **Raises**
    |   None

    | **References**
    |   None
    """
    return mean(sum((Y - X)**2, dim=-1))
