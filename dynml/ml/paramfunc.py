"""Define some parameterized function initializations and representations.

This module defines some parameterized function initializations and
representations.
"""


# import built-in python-package code
from abc import ABC, abstractmethod
from itertools import product
from math import prod
from typing import List, Tuple, Union
# import external python-package code
from torch import as_tensor, rand, Tensor, zeros
from torch.nn import Module, ModuleList, ParameterList
from torch.nn import Linear as TorchLinear
from torch.nn.functional import gelu, sigmoid
# import internal python-package code
# None


# export public code
__all__ = ["affine_init_torch",

           "whole_num_partition",
           "polynomial_init_uniform",

           "ParamFunc",
           "Identity", "CoefficientMult", "Add", "Flatten", "Unflatten",
           "Composition", "MatrixMult", "Polynomial", "Affine", "GELU",
           "Sigmoid", "FullyConnMLP"]


# define affine transformation initializations
def affine_init_torch(dim_in: int, dim_out: int) -> Tuple[Tensor, Tensor]:
    """Return a PyTorch initialized affine bias and matrix.

    This method returns a PyTorch initialized affine bias and matrix.

    | **Args**
    |   ``dim_in``: the input dimension
    |   ``dim_out``: the output dimension

    | **Return**
    |   ``Tuple[Tensor, Tensor]``: the PyTorch initialized affine bias with
            shape ``(1, dim_out)`` and affine matrix with shape
            ``(dim_in, dim_out)``
    | **Raises**
    |   None

    | **References**
    |   None
    """
    # initialize a PyTorch linear layer
    linear = TorchLinear(dim_in, dim_out)
    # return the affine bias and matrix
    return linear.bias.unsqueeze(0).detach(), linear.weight.T.detach()


# define polynomial initializations
def whole_num_partition(d: int, n: int) -> Tuple[Tuple[int, ...], ...]:
    """Return the permutations of ``n`` whole numbers that sum to ``d``.

    This method returns the permutations of ``n`` whole numbers that sum to
    ``d``. For example, the permutations of 3 whole numbers that sum to 2 are
    ``((2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1))``. We
    also guarantee that when ``d = 1`` the permutations follow the pattern
    ``((1, 0, 0), (0, 1, 0), (0, 0, 1))``, for example, when ``n = 3``.

    | **Args**
    |   ``d`` (``int``): the sum
    |   ``n`` (``int``): the number of whole numbers

    | **Return**
    |   ``Tuple[Tuple[int, ...]]``: the permutations of ``n`` whole numbers
            that sum to ``d``

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # find the permutations of n whole numbers that sum to d
    if d == 1:
        partition = tuple((0,) * i + (1,) + (0,) * (n - i - 1)
                          for i in range(n))
        return partition
    else:
        partition_list: List[Tuple[int, ...]] = []
        for whole_nums in product(range(d + 1), repeat=n):
            if sum(whole_nums) == d:
                partition_list.append(whole_nums)
        return tuple(partition_list)


def polynomial_init_uniform(dim_in: int, dim_out: int, bound: float,
                            degrees: Tuple[int, ...]) -> Tuple[Tensor, ...]:
    """Return polynomial coefficients sampled from the symmetric uniform dist..

    This method returns polynomial coefficients sampled from the symmetric
    uniform distribution: ``U[[-bound, bound)]``. For example, let
    ``dim_in = 3``, ``dim_out = 4``, and ``degrees = (2,)``. Say the output of
    ``whole_num_partition(degrees[0], dim_in)`` is ``powers = ((0, 0, 2),
    (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0))``. The output of
    this method is then a ``Tuple[Tensor, ...]`` with length
    ``len(degrees) = 1`` where the ``i``-th row of the ``Tensor``, with shape
    ``(dim_out,)``, corresponds to the polynomial coefficients associated with
    the ``i``-th power in ``powers``.

    | **Args**
    |   ``dim_in`` (``int``): the input dimension
    |   ``dim_out`` (``int``): the output dimension
    |   ``bound`` (``float``): the bound of the uniform distribution
    |   ``degrees`` (``Tuple[int, ...]``): the polynomial degrees

    | **Return**
    |   ``Tuple[Tensor, ...]``: the polynomial coefficients corresponding to
            the each ``degree`` in ``degrees``

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # sample the polynomial coefficients from the symmetric uniform dist.
    output: List[Tensor] = []
    for degree in degrees:
        num_powers = len(whole_num_partition(degree, dim_in))
        output.append(bound * (2 * rand(num_powers, dim_out) - 1))
    return tuple(output)


# define some parameterized functions
class ParamFunc(ABC, Module):
    """Represent a parameterized function.

    This abstract class represents a parameterized function.

    | **Abstract Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions

    | **Class Attributes**
    |   None

    | **Attributes**
    |   None

    | **Abstract Methods**
    |   ``forward()``: return the function's output given an input

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclasses
    |   ``__add__()``: return the addition of two parameterized functions
    |   ``__rmul__()``: return the multiplication of a parameterized function
            by a scalar
    |   ``__matmul__()``: return the matrix multiplication of two parameterized
            functions

    | **References**
    |   None
    """

    @property
    @abstractmethod
    def dims_in(self) -> Tuple[int, ...]:
        ...

    @property
    @abstractmethod
    def dims_out(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """Return the function's output given an input.

        This abstract method returns the function's output given an input.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the function's output with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    def __init__(self) -> None:
        """Initialize the superclasses.

        This method initializes the superclasses ``abc.ABC`` and
        ``torch.nn.Module``.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()

    def __add__(self, func2: "ParamFunc") -> "Add":
        """Return the addition of two parameterized functions.

        This method returns the addition of two parameterized functions.

        | **Args**
        |   ``func2`` (``ParamFunc``): the second parameterized function

        | **Return**
        |   ``Add``: the addition of the two parameterized functions

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the addition of the two parameterized functions
        return Add(self, func2)

    def __rmul__(self, coefficient: Union[float, complex]
                 ) -> "CoefficientMult":
        """Return the multiplication of a parameterized function by a scalar.

        This method returns the multiplication of a parameterized function by a
        scalar.

        | **Args**
        |   ``coefficient`` (``Union[float, complex]``): the scalar coefficient

        | **Return**
        |   ``CoefficientMult``: the scalar multiplication of the parameterized
                function

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the scalar multiplication of the parameterized function
        return CoefficientMult(coefficient, self)

    def __matmul__(self, func2: "ParamFunc") -> "MatrixMult":
        """Return the matrix multiplication of two parameterized functions.

        This method returns the matrix multiplication of two parameterized
        functions.

        | **Args**
        |   ``func2`` (``ParamFunc``): the second parameterized function

        | **Return**
        |   ``MatrixMult``: the matrix multiplication of the two parameterized
                functions

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the matrix multiplication of the two parameterized functions
        return MatrixMult(self, func2)


class Identity(ParamFunc):
    """Represent the identity function.

    This class represents the identity function.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: the identity function
    |   ``__init__()``: initialize the superclass
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
        return self._dims

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims

    def forward(self, x: Tensor) -> Tensor:
        """Return the input ``Tensor``.

        This method returns the input ``Tensor``.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the function's input with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the input Tensor
        return x

    def __init__(self, dims: Tuple[int, ...]) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``ParamFunc``.

        | **Args**
        |   ``dims`` (``Tuple[int, ...]``): the input and output dimensions

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # set the input and output dimension
        self._dims = dims


class CoefficientMult(ParamFunc):
    """Represent the multiplication of a parameterized function by a scalar.

    This class represents the multiplication of a parameterized function by a
    scalar.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``coefficient`` (``Union[float, complex]``): the scalar coefficient
    |   ``func`` (``ParamFunc``): the parameterized function

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the scalar mult. of the parameterized function's
            output
    |   ``__init__()``: initialize the superclass and attributes
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
        return self.func.dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self.func.dims_out

    def forward(self, x: Tensor) -> Tensor:
        """Return the scalar mult. of the parameterized function's output.

        This method returns the scalar mult. of the parameterized function's
        output.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the scalar multiplication of the parameterized
                function's output with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the scalar multiplication of the parameterized function output
        return self.coefficient * self.func.forward(x)

    def __init__(self, coefficient: Union[float, complex], func: ParamFunc
                 ) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass ``ParamFunc`` and the
        attributes.

        | **Args**
        |   ``coefficient`` (``Union[float, complex]``): the scalar coefficient
        |   ``func`` (``ParamFunc``): the parameterized function

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # set the scalar coefficient and parameterized function
        self.coefficient = coefficient
        self.func = func


class Add(ParamFunc):
    """Represent the addition of two parameterized functions.

    This class represents the addition of two parameterized functions.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``func1`` (``ParamFunc``): the first parameterized function
    |   ``func2`` (``ParamFunc``): the second parameterized function

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the sum of the two parameterized functions'
            output
    |   ``__init__()``: initialize the superclass and attributes
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
        return self.func1.dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self.func1.dims_out

    def forward(self, x: Tensor) -> Tensor:
        """Return the sum of the two parameterized functions' output.

        This method returns the sum of the two parameterized functions' output.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the sum of the two parameterized function's output
                with shape ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the sum of the two parameterized function's output
        return self.func1.forward(x) + self.func2.forward(x)

    def __init__(self, func1: ParamFunc, func2: ParamFunc) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass ``ParamFunc`` and the
        attributes.

        | **Args**
        |   ``func1`` (``ParamFunc``): the first parameterized function
        |   ``func2`` (``ParamFunc``): the second parameterized function

        | **Return**
        |   None

        | **Raises**
        |   ``ValueError``: if the input and output dimensions of the
                parameterized functions do not match

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # check the input and output dimensions of the parameterized functions
        cond1 = (func1.dims_in != func2.dims_in)
        cond2 = (func1.dims_out != func2.dims_out)
        if (cond1 or cond2):
            raise ValueError("The input and output dimensions of the "
                             + "parameterized functions do not match.")
        # set the parameterized functions
        self.func1 = func1
        self.func2 = func2


class Flatten(ParamFunc):
    """Represent the flattening of a ``Tensor``.

    This class represents the flattening of a ``Tensor``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the product of the input dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the flattened ``Tensor``
    |   ``__init__()``: initialize the superclass and attributes
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
        return (prod(self._dims_in),)

    def forward(self, x: Tensor) -> Tensor:
        """Return the flattened ``Tensor``.

        This method returns the flattened ``Tensor``.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the flattened tensor with shape
                ``(..., prod(self.dims_in))``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the flattened tensor
        return x.flatten(start_dim=-len(self.dims_in))

    def __init__(self, dims_in: Tuple[int, ...]) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass ``ParamFunc`` and attributes.

        | **Args**
        |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # set the dimensions to flatten
        self._dims_in = dims_in


class Unflatten(ParamFunc):
    """Represent the unflattening of a ``Tensor``.

    This class represents the unflattening of a ``Tensor``. This class is the
    inverse of ``Flatten`` when the input dimensions of ``Flatten`` are the
    same as the output dimensions of ``Unflatten``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the product of the output dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the unflattened ``Tensor``
    |   ``__init__()``: initialize the superclass and attributes
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
        return (prod(self._dims_out),)

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims_out

    def forward(self, x: Tensor) -> Tensor:
        """Return the unflattened ``Tensor``.

        This method returns the unflattened ``Tensor``.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(..., prod(self.dims_out))``

        | **Return**
        |   ``Tensor``: the unflattened tensor with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the unflattened tensor
        return x.unflatten(dim=-1, sizes=self.dims_out)

    def __init__(self, dims_out: Tuple[int, ...]) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass ``ParamFunc`` and attributes.

        | **Args**
        |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # set the dimensions to unflatten
        self._dims_out = dims_out


class Composition(ParamFunc):
    """Represent the composition of functions.

    This class represents the composition of functions.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``funcs`` (``ModuleList``): the list of functions to be
            composed where the first function is the first list element
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the composite function's output
    |   ``__init__()``: initialize the superclass and ``Tuple`` of composed
            functions
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
        return self.funcs[0].dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self.funcs[-1].dims_out

    def forward(self, x: Tensor) -> Tensor:
        """Return the composite function's output.

        This method returns the composite function's output.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the composite function's output with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # iterate over the functions
        for func in self.funcs:
            x = func.forward(x)
        return x

    def __init__(self, funcs: Tuple[ParamFunc, ...]) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``ParamFunc``.

        | **Args**
        |   ``funcs`` (``Tuple[ParamFunc, ...]``): the list of composed
                functions

        | **Return**
        |   None

        | **Raises**
        |   ``ValueError``: if the input and output dimensions of the composed
                functions do not match

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # check the input and output dimensions of the composed functions
        dims_out = funcs[0].dims_in
        for func in funcs:
            if dims_out != func.dims_in:
                raise ValueError("The input and output dimensions of the "
                                 + "composed functions do not match.")
            dims_out = func.dims_out
        # set the tuple of composed functions
        self.funcs = ModuleList(funcs)


class MatrixMult(ParamFunc):
    """Represent the matrix multiplication of two parameterized functions.

    This class represents the matrix multiplication of two parameterized
    functions.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``func1`` (``ParamFunc``): the first parameterized function
    |   ``func2`` (``ParamFunc``): the second parameterized function

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the matrix mult. of the two
            parameterized functions' output
    |   ``__init__()``: initialize the superclass and attributes
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
        return self.func1.dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self.func1.dims_out[:-1] + self.func2.dims_out[1:]

    def forward(self, x: Tensor) -> Tensor:
        """Return the matrix mult. of the two parameterized functions' output.

        This method returns the matrix multiplication of the two parameterized
        functions' output.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the matrix multiplication of the two parameterized
                functions' output with shape ``(...,) +
                self.func1.dims_out[:-1] + self.func2.dims_out[1:]``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the matrix multiplication of the two parameterized functions'
        # output
        return self._forward(self.func1.forward(x), self.func2.forward(x))

    def __init__(self, func1: ParamFunc, func2: ParamFunc) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass ``ParamFunc`` and the
        attributes.

        | **Args**
        |   ``func1`` (``ParamFunc``): the first parameterized function
        |   ``func2`` (``ParamFunc``): the second parameterized function

        | **Return**
        |   None

        | **Raises**
        |   ``ValueError``: if the input dimensions of the parameterized
                functions are not the same
        |   ``ValueError``: if the output dimensions of the parameterized
                functions are not compatible
        |   ``ValueError``: if the second function does not have a 2D output
        |   ``ValueError``: if the first function has greater than 2D output

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # check the input dimensions of the parameterized functions
        if (func1.dims_in != func2.dims_in):
            raise ValueError("The input dimensions of the parameterized "
                             + "functions are not the same.")
        # check the input and output dimensions of the parameterized functions
        if (func1.dims_out[-1] != func2.dims_out[-len(func2.dims_out)]):
            raise ValueError("The output dimensions of the parameterized "
                             + "functions are not compatible.")
        # check the second function has a 2D output
        if len(func2.dims_out) != 2:
            raise ValueError("The second function does not have a 2D output.")
        # check the first function has greater than 2D output
        if len(func1.dims_out) > 2:
            raise ValueError("The first function has greater than 2D output.")
        # set the parameterized functions
        self.func1 = func1
        self.func2 = func2
        # set the forward method
        if len(func1.dims_out) == 1:
            self._forward = self._vec_mat_forward
        else:
            self._forward = self._mat_mat_forward

    def _vec_mat_forward(self, x: Tensor, y: Tensor) -> Tensor:
        return (x.unsqueeze(-2) @ y).squeeze(-2)

    def _mat_mat_forward(self, x: Tensor, y: Tensor) -> Tensor:
        return x @ y


class Polynomial(ParamFunc):
    """Represent a polynomial function between vector spaces.

    This class represents a polynomial function between vector spaces.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``degrees`` (``Tuple[int, ...]``): the polynomial degrees
    |   ``coefficients`` (``ParameterList``): the polynomial coefficient
            tensors

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass and attributes
    |   ``forward()``: return the polynomial function's output
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
        return (self._dim_in,)

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return (self._dim_out,)

    def __init__(self, dim_in: int, dim_out: int,
                 degrees: Tuple[int, ...], coefficients: Tuple[Tensor, ...]
                 ) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass ``ParamFunc`` and the
        attributes. For example, let ``dim_in = 3``, ``dim_out = 4``, and
        ``degrees = (2,)``. Say the output of ``whole_num_partition(degree[0],
        dim_in)`` is ``powers = ((0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1),
        (1, 1, 0), (2, 0, 0))``. The ``coefficients`` argument is then a
        ``Tuple[Tensor, ...]`` with length ``len(degrees) = 1`` where the
        ``i``-th row of the ``Tensor``, with shape ``(dim_out,)``,
        corresponds to the polynomial coefficients associated with the ``i``-th
        power in ``powers``.

        | **Args**
        |   ``degrees`` (``Tuple[int, ...]``): the polynomial degrees
        |   ``coefficients`` (``Tuple[Tensor, ...]``): the polynomial
                coefficient tensors

        | **Return**
        |   None

        | **Raises**
        |   ``ValueError``: if the number of polynomial degrees and
                coefficients do not match
        |   ``ValueError``: if the coefficient tensors are not the right shape

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # check the number of polynomial degrees and coefficients match
        if len(degrees) != len(coefficients):
            raise ValueError("The number of polynomial degrees and "
                             + "coefficients do not match.")
        # check the coefficient tensors are the right shape
        self._powers_list: List[Tuple[Tuple[int, ...], ...]] = []
        for degree, coefficient in zip(degrees, coefficients):
            powers = whole_num_partition(degree, dim_in)
            self._powers_list.append(powers)
            num_powers = len(powers)
            if coefficient.shape != (num_powers, dim_out):
                raise ValueError("The coefficient tensor is not the right "
                                 + "shape.")
        # set dimension attributes
        self._dim_in = dim_in
        self._dim_out = dim_out
        # set the polynomial degrees and coefficients
        self.degrees = degrees
        self.coefficients = ParameterList(coefficient
                                          for coefficient in coefficients)

    def forward(self, x: Tensor) -> Tensor:
        """Return the polynomial function's output.

        This method returns the polynomial function's output.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the polynomial function's output with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the polynomial function's output
        device = next(self.parameters()).device.type
        output = zeros(x.shape[:-1] + self.dims_out, device=device)
        for powers, coefficient in zip(self._powers_list, self.coefficients):
            for i, power in enumerate(powers):
                power_tensor = as_tensor(power, device=device)
                output += ((x ** power_tensor).prod(dim=-1, keepdim=True)
                           * coefficient[i])
        return output


class Affine(Polynomial):
    """Represent an affine transformation.

    This class represents an affine transformation.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``degrees`` (``Tuple[int, ...]``): the polynomial degrees
    |   ``coefficients`` (``Tuple[Tensor, ...]``): the polynomial coefficient
            tensors

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass
    |   ``forward()``: return the affine transformation's output
    |   ``__add__()``: return the addition of two parameterized functions
    |   ``__rmul__()``: return the multiplication of a parameterized function
            by a scalar
    |   ``__matmul__()``: return the matrix multiplication of two parameterized
            functions

    | **References**
    |   None
    """

    def __init__(self, dim_in: int, dim_out: int, b_T: Union[Tensor, None],
                 A_T: Union[Tensor, None]) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``Polynomial``.

        | **Args**
        |   ``b_T`` (``Union[Tensor, None]``): the bias with shape
                ``(1, dim_out)``
        |   ``A_T`` (``Union[Tensor, None]``): the matrix with shape
                ``(dim_in, dim_out)``

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        if (A_T is not None) and (b_T is not None):
            super().__init__(dim_in, dim_out, (0, 1), (b_T, A_T))
            self._forward = self._forward_bA
        elif (A_T is not None) and (b_T is None):
            super().__init__(dim_in, dim_out, (1,), (A_T,))
            self._forward = self._forward_A
        elif (A_T is None) and (b_T is not None):
            super().__init__(dim_in, dim_out, (0,), (b_T,))
            self._forward = self._forward_b
        elif (A_T is None) and (b_T is None):
            raise ValueError("The bias and matrix cannot both be None.")

    def forward(self, x: Tensor) -> Tensor:
        """Return the affine transformation's output.

        This method returns the affine transformation's output.

        | **Args**
        |   ``x`` (``Tensor``): an input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the affine transformation's output with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the affine transformation's output
        return self._forward(x)

    def _forward_bA(self, x: Tensor) -> Tensor:
        return self.coefficients[0] + x @ self.coefficients[1]

    def _forward_A(self, x: Tensor) -> Tensor:
        return x @ self.coefficients[0]

    def _forward_b(self, x: Tensor) -> Tensor:
        return self.coefficients[0]


class GELU(ParamFunc):
    """Represent the Gaussian Error Linear Unit (G.E.L.U.) activation layer.

    This class represents the G.E.L.U. activation layer.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``int``): the input dimensions
    |   ``dims_out`` (``int``): the output dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the G.E.L.U. activation function's output
    |   ``__init__()``: initialize the superclass

    | **References**
    |   [1] Hendrycks, D. and Gimpel, K., 2016. Gaussian Error Linear Units
            (GeLUs). arXiv preprint arXiv:1606.08415.
    """

    @property
    def dims_in(self) -> Tuple[int, ...]:
        return self._dims

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims

    def forward(self, x: Tensor) -> Tensor:
        """Return the G.E.L.U. activation function's output.

        This method returns the G.E.L.U. activation function's output. In
        particular, this method returns the G.E.L.U. activation function
        element-wise applied to the input.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the G.E.L.U. activation function's output with shape
                ``(...,) + self.dims_in``

        | **Raises**
        |   None

        | **References**
        |   [1] Hendrycks, D. and Gimpel, K., 2016. Gaussian Error Linear Units
                (GeLUs). arXiv preprint arXiv:1606.08415.
        """
        # return the G.E.L.U. activation function's output
        return gelu(x)

    def __init__(self, dims: Tuple[int, ...]) -> None:
        """Initialize the superclass.

        This method initializes the superclass.

        | **Args**
        |   ``dims`` (``Tuple[int, ...]``): the input and output dimensions

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass constructor
        super().__init__()
        # initialize the dimension attributes
        self._dims = dims


class Sigmoid(ParamFunc):
    """Represent the sigmoid activation layer.

    This class represents the sigmoid activation layer.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``forward()``: return the sigmoid activation function's output
    |   ``__init__()``: initialize the superclass and attributes
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
        return self._dims

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims

    def forward(self, x: Tensor) -> Tensor:
        """Return the sigmoid activation function's output.

        This method returns the sigmoid activation function's output. In
        particular, this method returns the sigmoid activation function
        element-wise applied to the input.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the sigmoid activation function's output with shape
                ``(...,) + self.dims_in``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return sigmoid(x)

    def __init__(self, dims: Tuple[int, ...]) -> None:
        """Initialize the superclass and attributes.

        This method initializes the superclass and attributes.

        | **Args**
        |   ``dims`` (``Tuple[int, ...]``): the input and output dimensions

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        super().__init__()
        self._dims = dims


class FullyConnMLP(ParamFunc):
    """Represent a fully connected M.L.P..

    This class represents a fully connected M.L.P..

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``TYPES`` (``Dict[str, Tuple[Type[ParamFunc],
            Callable[[int, int], Tuple[Tensor, Tensor]]]``): the dictionary
            mapping the M.L.P. types to the activation layers and affine
            initialization functions

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
    |   ``layers`` (``Composition``): the composition M.L.P. layers

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``types()``: return the list of M.L.P. types

    | **Methods**
    |   ``forward()``: return the M.L.P.'s output
    |   ``__init__()``: initializes the superclass and the M.L.P.'s attributes

    | **References**
    |   None
    """

    TYPES = {'gelu_torch': (GELU, affine_init_torch),
             'sigmoid_torch': (Sigmoid, affine_init_torch)}

    @property
    def dims_in(self) -> Tuple[int, ...]:
        return self._dims_in

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return self._dims_out

    def forward(self, x: Tensor) -> Tensor:
        """Returns the M.L.P.'s output.

        This method returns the M.L.P.'s output.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape
                ``(...,) + self.dims_in``

        | **Return**
        |   ``Tensor``: the output with shape
                ``(...,) + self.dims_out``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # pass the input tensor through the layers iteratively
        return self.layers.forward(x)

    def __init__(self, dims_in: Tuple[int, ...], dims_hidden: Tuple[int, ...],
                 dims_out: Tuple[int, ...], type: str) -> None:
        """Initializes the superclass and the M.L.P.'s attributes.

        This method initializes the superclass and the M.L.P.'s attributes.
        Using the ``type`` argument, one can choose the M.L.P. type from the
        tuple ``('gelu_torch', 'sigmoid_torch')``. The ``type='gelu_torch'``
        option initializes
        the M.L.P. with G.E.L.U. activation layers and
        ``torch.nn.Linear``-default initialized affine layers. The
        ``type='sigmoid_torch'`` option initializes the M.L.P. with sigmoid
        activation layers and ``torch.nn.Linear``-default initialized affine
        layers.

        | **Args**
        |   ``dims_in`` (``Tuple[int, ...]``): the input dimensions
        |   ``dims_hidden`` (``Tuple[int, ...]``): the hidden layer dimensions
        |   ``dims_out`` (``Tuple[int, ...]``): the output dimensions
        |   ``type`` (``str``): the M.L.P. type from the tuple
                ``('gelu_torch', 'sigmoid_torch')``

        | **Return**
        |   None

        | **Raises**
        |   ``ValueError``: if the M.L.P. type is unknown

        | **References**
        |   None
        """
        # initialize the super class
        super().__init__()
        # initialize the input and output dimensions
        self._dims_in = dims_in
        self._dims_out = dims_out
        # initialize the layers
        if type in FullyConnMLP.TYPES:
            activ = FullyConnMLP.TYPES[type][0]
            init = FullyConnMLP.TYPES[type][1]
        else:
            raise ValueError("The M.L.P. type is unknown.")
        layers_list: List[ParamFunc] = list()
        layers_list.append(Flatten(self.dims_in))
        layers_list.append(Affine(prod(self.dims_in), dims_hidden[0],
                                  *init(prod(self._dims_in), dims_hidden[0])))
        layers_list.append(activ((dims_hidden[0],)))
        for i in range(1, len(dims_hidden)):
            layers_list.append(Affine(dims_hidden[i - 1], dims_hidden[i],
                                      *init(dims_hidden[i - 1],
                                            dims_hidden[i])))
            layers_list.append(activ((dims_hidden[i],)))
        layers_list.append(Affine(dims_hidden[-1], prod(self.dims_out),
                                  None, init(dims_hidden[-1],
                                             prod(self.dims_out))[1]))
        layers_list.append(Unflatten(self.dims_out))
        self.layers = Composition(tuple(layers_list))

    @classmethod
    def types(cls) -> List[str]:
        """Return the list of M.L.P. types.

        This method returns the list of M.L.P. types.

        | **Args**
        |   None

        | **Return**
        |   ``List[str]``: the list of M.L.P. types

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return list(cls.TYPES.keys())
