"""Define all functions and classes related to dimensionality reduction.

This module defines all functions and classes related to dimensionality
reduction.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import Tensor
from torch.linalg import norm, svd
# import internal python-package code
from dynml.ml.paramfunc import Composition, ParamFunc


# export public code
__all__ = ["local_pca",

           "DimRed",
           "DimRedComposition"]


# define dimensionality reduction functions
def local_pca(X: Tensor, center: Tensor, r: float) -> Tuple[Tensor, Tensor]:
    """Return the local P.C.A. modes and singular values.

    This method returns the local P.C.A. modes and singular values.

    | **Args**
    |   ``X`` (``Tensor``): the data set with shape ``(...,) + (dim,)``
    |   ``center`` (``Tensor``): the local P.C.A. center with shape ``(dim,)``
    |   ``r`` (``float``): the local P.C.A. radius

    | **Returns**
    |   ``Tuple[Tensor, Tensor]``: the local P.C.A. modes ``Phi_T`` with shape
            ``(dim, dim)`` and the singular values ``s`` with shape ``(dim,)``

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # compute local P.C.A. modes and singular values
    X_data = X.flatten(start_dim=0, end_dim=-2) - center
    num_samples = len(X_data)
    X_data = X_data[norm(X_data, dim=1) < r]
    Phi, s, _ = svd(X_data.T @ X_data / (num_samples - 1), full_matrices=False)
    return Phi.T, s


# define dimensionality reduction classes
class DimRed(ParamFunc):
    """Represent a dimensionality reducer.

    This class represents a dimensionality reducer.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``enc`` (``ParamFunc``): the encoder
    |   ``dec`` (``ParamFunc``): the decoder
    |   ``dims_latent`` (``Tuple[int, ...]``): the latent dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the super class and the attributes
    |   ``forward()``: return the input after encoding and decoding
    |   ``__add__()``: return the addition of two parameterized functions
    |   ``__rmul__()``: return the multiplication of a parameterized function
            by a scalar
    |   ``__matmul__()``: return the matrix multiplication of two parameterized
            functions

    | **References**
    |   None
    """

    @property
    def dims_in(self) -> Tuple[int, ...]:
        return self._dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims_out

    def __init__(self, enc: ParamFunc, dec: ParamFunc) -> None:
        """Initialize the super class and the attributes.

        This method initializes the super class and the attributes.

        | **Args**
        |   ``enc`` (``ParamFunc``): the encoder
        |   ``dec`` (``ParamFunc``): the decoder

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize super class
        super().__init__()
        # initialize attributes
        self.enc = enc
        self.dec = dec
        self._forward = Composition((enc, dec))
        self._dims_in = enc.dims_in
        self.dims_latent = enc.dims_out
        self._dims_out = dec.dims_out

    def forward(self, x: Tensor) -> Tensor:
        """Return the input after encoding and decoding.

        This method returns the input after encoding and decoding.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape ``(...,) + (dims_in,)``

        | **Returns**
        |   ``Tensor``: the output with shape ``(...,) + (dims_out,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return self._forward(x)


class DimRedComposition(DimRed):
    """Represent a composition of dimensionality reducers.

    This class represents a composition of dimensionality reducers.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``enc`` (``ParamFunc``): the encoder
    |   ``dec`` (``ParamFunc``): the decoder
    |   ``dims_latent`` (``Tuple[int, ...]``): the latent dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the super class and the attributes
    |   ``forward()``: return the input after encoding and decoding
    |   ``__add__()``: return the addition of two parameterized functions
    |   ``__rmul__()``: return the multiplication of a parameterized function
            by a scalar
    |   ``__matmul__()``: return the matrix multiplication of two parameterized
            functions

    | **References**
    |   None
    """

    @property
    def dims_in(self) -> Tuple[int, ...]:
        return self._dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims_out

    def __init__(self, dimreds: Tuple[DimRed, ...]) -> None:
        """Initialize the super class and the attributes.

        This method initializes the super class and the attributes.

        | **Args**
        |   ``dimreds`` (``Tuple[DimRed]``): the dimensionality reducers

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        enc = Composition(tuple(reducer.enc for reducer in dimreds))
        dec = Composition(tuple(reducer.dec for reducer in reversed(dimreds)))
        super().__init__(enc, dec)
