"""Define the computational-environment configuration function.

This module defines the computational-environment configuration function.
"""


# import built-in python-package code
from random import seed as python_seed
# import external python-package code
from torch import float16, float32, float64, set_default_dtype
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available
from torch.cuda import manual_seed as cuda_manual_seed
# import internal python-package code
# None


# export public code
__all__ = ["config"]


# define test setup utility function
def config(b: int, s: int) -> str:
    """Configure the computational environment.

    This function configures the numerical data type, the random seed, and
    returns 'cuda' if available, otherwise the function returns 'cpu'.

    | **Args**
    |   ``b`` (``int``): the numerical data type precision in bits, which can
    |   be either ``16``, ``32``, or ``64``.
    |   ``s`` (``int``): the random seed

    | **Return**
    |   ``str``: the device, which can be either ``"cuda"`` or ``"cpu"``

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # configure the numerical default data type for python
    # NA
    # configure the numerical default data type for sympy
    # NA
    # configure the numerical default data type for torch
    if b == 16:
        set_default_dtype(float16)
    elif b == 32:
        set_default_dtype(float32)
    elif b == 64:
        set_default_dtype(float64)
    # configure the random seed for python
    python_seed(s)
    # configure the random seed for sympy
    # NA
    # configure the random seed for torch
    torch_manual_seed(s)
    cuda_manual_seed(s)
    # return the device
    return 'cuda' if is_available() else 'cpu'
