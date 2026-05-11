"""Test the ``dynml.config`` module.

This module tests the ``dynml.config`` module.
"""


# import built-in python-package code
from random import randint
from random import seed as set_seed
# import external python-package code
from torch import float16, float32, float64, rand, tensor
from torch.random import manual_seed as torch_manual_seed
from torch.cuda import is_available
from torch.cuda import manual_seed as cuda_manual_seed
# import internal python-package code
from dynml.utils.config import config


# test setup
def test_config() -> None:
    """Test the ``config`` function.

    This function tests the ``config`` function.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # test setup
    bs = [16, 32, 64]
    fs = [float16, float32, float64]
    s = 0
    for b, f in zip(bs, fs):
        # device is configured for python
        # NA
        # device is configured for sympy
        # NA
        # device is configured for torch
        test_g_t = config(b, s)
        desired_g_t = 'cuda' if is_available() else 'cpu'
        assert test_g_t == desired_g_t
        device = desired_g_t
        # dtype is configured for python
        # NA
        # dtype is configured for sympy
        # NA
        # dtype is configured for torch
        config(b, s)
        test_d_t = tensor(1., device=device).dtype
        desired_d_t = f
        assert test_d_t == desired_d_t
        # seed is configured for python
        config(b, s)
        test_s_p = randint(0, 100)
        set_seed(0)
        desired_s_p = randint(0, 100)
        assert test_s_p == desired_s_p
        # seed is configured for sympy
        # NA
        # seed is configured for torch
        config(b, s)
        test_s_t = rand(2, device=device)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        desired_s_t = rand(2, device=device)
        assert test_s_t.allclose(desired_s_t, atol=0.0)
