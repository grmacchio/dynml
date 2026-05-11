"""Test the ``dynml.ml.loss`` module.

This module tests the ``dynml.ml.loss`` module.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import mean, tensor
from torch.linalg import norm
# import internal python-package code
from dynml.ml.loss import recon_loss
from dynml.utils.config import config


# test recon_loss()
def test_recon_loss():
    """Test the ``recon_loss()`` method.

    This method tests the ``recon_loss()`` method.

    | **Args**
    |   None

    | **Returns**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # configure the test
    device = config(64, 0)
    # test recon_loss()
    y = tensor([[1., 2.], [3., 4.], [5., 6.]], device=device)
    x = tensor([[7., 8.], [9., 10.], [11., 12.]], device=device)
    desired = mean(norm(y - x, dim=-1)**2)
    assert desired.allclose(recon_loss(y, x))
