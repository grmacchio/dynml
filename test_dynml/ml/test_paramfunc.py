"""Test the ``dynml.ml.paramfunc`` module.

This module tests the ``dynml.ml.paramfunc`` module.
"""


# import built-in python-package code
from typing import Tuple
from random import seed as python_seed
# import external python-package code
from torch import erf, exp, float64, rand, set_default_dtype, sqrt, stack
from torch import tensor, Tensor
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available
from torch.cuda import manual_seed as cuda_manual_seed
from torch.nn import Linear as TorchLinear
from torch.nn.functional import gelu
# import internal python-package code
from dynml.ml.paramfunc import affine_init_torch, whole_num_partition
from dynml.ml.paramfunc import polynomial_init_uniform, ParamFunc, Identity
from dynml.ml.paramfunc import CoefficientMult, Add, Flatten, Unflatten
from dynml.ml.paramfunc import Composition, MatrixMult, Polynomial, Affine
from dynml.ml.paramfunc import GELU, Sigmoid, FullyConnMLP


# test affine_init_torch
def test_affine_init_torch() -> None:
    """Test the ``affine_init_torch()`` method.

    This method tests the ``affine_init_torch()`` method.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # test the affine_init_torch() method
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    b, A = affine_init_torch(3, 2)
    assert A.shape == (3, 2)
    assert b.shape == (1, 2)
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    torchlinear = TorchLinear(3, 2)
    assert A.detach().allclose(torchlinear.weight.T.detach(), atol=0.0)
    assert b.detach().allclose(torchlinear.bias.unsqueeze(0).detach(),
                               atol=0.0)


# test whole_num_partition
def test_whole_num_partition() -> None:
    """Test the ``whole_num_partition()`` method.

    This method tests the ``whole_num_partition()`` method.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # test the whole_num_partition() method
    d = 2
    n = 3
    desired1: Tuple[Tuple[int, ...], ...] = ((2, 0, 0), (0, 2, 0), (0, 0, 2),
                                             (1, 1, 0), (1, 0, 1), (0, 1, 1))
    for partition in desired1:
        assert partition in whole_num_partition(d, n)
    d = 1
    n = 4
    desired2: Tuple[Tuple[int, ...], ...] = ((1, 0, 0, 0), (0, 1, 0, 0),
                                             (0, 0, 1, 0), (0, 0, 0, 1))
    for i, partition in enumerate(desired2):
        assert partition == whole_num_partition(d, n)[i]
    d = 0
    n = 3
    desired3: Tuple[Tuple[int, ...], ...] = ((0, 0, 0),)
    for partition in desired3:
        assert partition in whole_num_partition(d, n)


# test polynomial_init_uniform
def test_polynomial_init_uniform() -> None:
    """Test the ``polynomial_init_uniform()`` method.

    This method tests the ``polynomial_init_uniform()`` method.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # test the polynomial_init_uniform() method
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    degrees = (0, 1, 2)
    c1, c2, c3 = polynomial_init_uniform(3, 2, 2.0, degrees)
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    powers: Tuple[Tuple[int, ...], ...] = ((0, 0, 0),)
    desired = 2.0 * (2 * rand(len(powers), 2) - 1)
    assert c1.allclose(desired, atol=0.0)
    powers = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    desired = 2.0 * (2 * rand(len(powers), 2) - 1)
    assert c2.allclose(desired, atol=0.0)
    powers = ((2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2))
    desired = 2.0 * (2 * rand(len(powers), 2) - 1)
    assert c3.allclose(desired, atol=0.0)


# test ParamFunc
class ParamFuncExample(ParamFunc):
    """Represent an example subclass of ``ParamFunc``.

    This class represents an example subclass of ``ParamFunc``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_in`` (``Tuple[int, ...]``): the input dimension of (2,)
    |   ``dims_out`` (``Tuple[int, ...]``): the output dimension of (2,)

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass
    |   ``forward()``: the identity function
    |   ``d_forward()``: return the input and input vector
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
        return (2,)

    @property
    def dims_out(self) -> Tuple[int, ...]:
        return (2,)

    def __init__(self) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``ParamFunc``.

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


def test_ParamFunc() -> None:
    """Test the ``ParamFunc`` abstract class.

    This method tests the ``ParamFunc`` abstract class. In particular,
    this method instantiates ``ParamFuncExample`` and tests
    ``dims_in``, ``dims_out``, ``forward()``, and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``. The methods ``__add__()``,
    ``__rmul__()``, and ``__matmul__()`` are tested in ``test_Add()``,
    ``test_CoefficientMult()``, and ``test_MatrixMult()``, respectively.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of ParamFuncExample
    test1 = ParamFuncExample().to(device)
    # test dims_in
    assert test1.dims_in == (2,)
    # test dims_out
    assert test1.dims_out == (2,)
    # test forward()
    x = tensor([1., 2.], device=device)
    desired = x
    assert test1.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([1., 2.], device=device, requires_grad=True)
    y = test1.forward(x)
    y.sum().backward()


# test Identity
def test_Identity() -> None:
    """Test the ``Identity`` class.

    This method tests the ``Identity`` class.  In particular, this
    instantiates ``Identity`` and tests ``dims_in``, ``dims_out``,
    ``forward()``, and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of Identity
    test = Identity((2,)).to(device)
    # test dims_in
    assert test.dims_in == (2,)
    # test dims_out
    assert test.dims_out == (2,)
    # test forward()
    x = tensor([1., 2.], device=device)
    desired = x
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([1., 2.], device=device, requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test CoefficientMult
def test_CoefficientMult() -> None:
    """Test the ``CoefficientMult`` class.

    This method tests the ``CoefficientMult`` class.  In particular, this
    instantiates ``CoefficientMult`` and tests ``dims_in``, ``dims_out``,
    ``coefficient``, ``func``, ``forward()``, and
    ``torch.nn.Module`` backward differentiation with respect to ``forward()``.
    Finally, we test the ``__rmul__()`` method associated with ``ParamFunc``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate CoefficientMult
    test = CoefficientMult(2.0, Identity((2,))).to(device)
    # test dims_in
    assert test.dims_in == (2,)
    # test dims_out
    assert test.dims_out == (2,)
    # test coefficient
    assert test.coefficient == 2.0
    # test func
    assert isinstance(test.func, Identity)
    # test forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = 2.0 * x
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device,
               requires_grad=True)
    y = test.forward(x)
    y.sum().backward()
    # test __rmul__()
    test3 = (2.0 * Identity((2,))).to(device)
    assert isinstance(test3, CoefficientMult)
    assert test3.dims_in == (2,)
    assert test3.dims_out == (2,)
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = 2.0 * x
    assert test3.forward(x).allclose(desired, atol=0.0)


# test Add
def test_Add() -> None:
    """Test the ``Add`` class.

    This method tests the ``Add`` class.  In particular, this instantiates
    ``Add`` and tests ``dims_in``, ``dims_out``, ``forward()``,
    and ``torch.nn.Module`` backward differentiation with
    respect to ``forward()``. Finally, we test the ``__add__()`` method
    associated with ``ParamFunc``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate Add
    try:
        Add(2 * Identity((2,)), 3 * Identity((3,)))
        raise NotImplementedError("Test not implemented")
    except ValueError as error:
        assert str(error) == ("The input and output dimensions of the "
                              + "parameterized functions do not match.")
    test = Add(2 * Identity((2,)), 3 * Identity((2,))).to(device)
    # test dims_in
    assert test.dims_in == (2,)
    # test dims_out
    assert test.dims_out == (2,)
    # test forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = 2.0 * x + 3.0 * x
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device,
               requires_grad=True)
    y = test.forward(x)
    y.sum().backward()
    # test __add__()
    test2 = (2 * Identity((2,)) + 3 * Identity((2,))).to(device)
    assert isinstance(test2, Add)
    assert test2.dims_in == (2,)
    assert test2.dims_out == (2,)
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = 2.0 * x + 3.0 * x
    assert test2.forward(x).allclose(desired, atol=0.0)


# test Flatten
def test_Flatten() -> None:
    """Test the ``Flatten`` class.

    This method tests the ``Flatten`` class.  In particular, this
    instantiates ``Flatten`` and tests ``dims_in``, ``dims_out``,
    ``forward()``, and ``torch.nn.Module`` backward differentiation with
    respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the Flatten
    test = Flatten((2, 3)).to(device)
    # test dims_in
    assert test.dims_in == (2, 3)
    # test dims_out
    assert test.dims_out == (6,)
    # test forward()
    x = tensor([[[1., 2., 3.],
                 [4., 5., 6.]],
                [[7., 8., 9.],
                 [10., 11., 12.]]], device=device)
    desired = tensor([[1., 2., 3., 4., 5., 6.],
                      [7., 8., 9., 10., 11., 12.]], device=device)
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[[1., 2., 3.],
                 [4., 5., 6.]],
                [[7., 8., 9.],
                 [10., 11., 12.]]], device=device, requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test Unflatten
def test_Unflatten() -> None:
    """Test the ``Unflatten`` class.

    This method tests the ``Unflatten`` class.  In particular, this
    instantiates ``Unflatten`` and tests ``dims_in``, ``dims_out``,
    ``forward()``, and ``torch.nn.Module`` backward differentiation with
    respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the Unflatten
    test = Unflatten((2, 3)).to(device)
    # test dims_in
    assert test.dims_in == (6,)
    # test dims_out
    assert test.dims_out == (2, 3)
    # test forward()
    x = tensor([[1., 2., 3., 4., 5., 6.],
                [7., 8., 9., 10., 11., 12.]], device=device)
    desired = tensor([[[1., 2., 3.],
                       [4., 5., 6.]],
                      [[7., 8., 9.],
                       [10., 11., 12.]]], device=device)
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1., 2., 3., 4., 5., 6.],
                [7., 8., 9., 10., 11., 12.]], device=device,
               requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test Composition
def test_Composition() -> None:
    """Test the ``Composition`` class.

    This method tests the ``Composition`` class.  In particular, this
    instantiates ``Composition`` and tests ``dims_in``, ``dims_out``,
    ``funcs``, ``forward()``, and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the Composition
    try:
        Composition((Identity((2,)), Identity((3,))))
        raise NotImplementedError("Test not implemented")
    except ValueError as error:
        assert str(error) == ("The input and output dimensions of the "
                              + "composed functions do not match.")
    test = Composition((2 * Identity((2,)), 3 * Identity((2,)))).to(device)
    # test dims_in
    assert test.dims_in == (2,)
    # test dims_out
    assert test.dims_out == (2,)
    # test funcs
    assert isinstance(test.funcs[0], CoefficientMult)
    assert isinstance(test.funcs[1], CoefficientMult)
    # test forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = 2.0 * 3.0 * x
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device,
               requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test MatrixMult
def test_MatrixMult() -> None:
    """Test the ``MatrixMult`` class.

    This method tests the ``MatrixMult`` class.  In particular, this
    instantiates ``MatrixMult`` and tests ``dims_in``, ``dims_out``,
    ``func1``, ``func2``, ``forward()`` and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``. Finally, we test the
    ``__matmul__()`` method associated with ``ParamFunc``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate MatrixMult
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    try:
        func1 = Composition((Identity((12,)), Unflatten((4, 3))))
        func2 = Composition((Identity((9,)), Unflatten((3, 3))))
        MatrixMult(func1, func2)
        raise NotImplementedError("Test not implemented")
    except ValueError as error:
        assert str(error) == ("The input dimensions of the parameterized "
                              + "functions are not the same.")
    try:
        func1 = Composition((Identity((12,)), Unflatten((4, 3))))
        func2 = Composition((Identity((12,)), Unflatten((4, 3))))
        MatrixMult(func1, func2)
        raise NotImplementedError("Test not implemented")
    except ValueError as error:
        assert str(error) == ("The output dimensions of the parameterized "
                              + "functions are not compatible.")
    try:
        func1 = Composition((Identity((12,)), Unflatten((4, 3))))
        func2 = Composition((Identity((12,)), Unflatten((3, 2, 2))))
        MatrixMult(func1, func2)
        raise NotImplementedError("Test not implemented")
    except ValueError as error:
        assert str(error) == ("The second function does not have a 2D output.")
    try:
        func1 = Composition((Identity((12,)), Unflatten((2, 2, 3))))
        func2 = Composition((Identity((12,)), Unflatten((3, 4))))
        MatrixMult(func1, func2)
        raise NotImplementedError("Test not implemented")
    except ValueError as error:
        assert str(error) == ("The first function has greater than 2D output.")
    func1 = Composition((Identity((12,)), Unflatten((4, 3))))
    func2 = Composition((Identity((12,)), Unflatten((3, 4))))
    test = MatrixMult(func1, func2).to(device)
    # test dims_in
    assert test.dims_in == (12,)
    # test dims_out
    assert test.dims_out == (4, 4)
    # test func1
    assert isinstance(test.func1, Composition)
    # test func2
    assert isinstance(test.func2, Composition)
    # test forward()
    x = tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                 [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]],
                [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]]],
               device=device)
    output1 = func1.forward(x)
    output2 = func2.forward(x)
    matrix00 = output1[0, 0] @ output2[0, 0]
    matrix01 = output1[0, 1] @ output2[0, 1]
    matrix10 = output1[1, 0] @ output2[1, 0]
    matrix11 = output1[1, 1] @ output2[1, 1]
    output00 = test.forward(x)[0, 0]
    output01 = test.forward(x)[0, 1]
    output10 = test.forward(x)[1, 0]
    output11 = test.forward(x)[1, 1]
    assert output00.allclose(matrix00, atol=0.0)
    assert output01.allclose(matrix01, atol=0.0)
    assert output10.allclose(matrix10, atol=0.0)
    assert output11.allclose(matrix11, atol=0.0)
    Func1 = Affine(12, 3, *affine_init_torch(12, 3)).to(device)
    Func2 = Composition((Identity((12,)), Unflatten((3, 4)))).to(device)
    output1 = Func1.forward(x)
    output2 = Func2.forward(x)
    matrix00 = output1[0, 0] @ output2[0, 0]
    matrix01 = output1[0, 1] @ output2[0, 1]
    matrix10 = output1[1, 0] @ output2[1, 0]
    matrix11 = output1[1, 1] @ output2[1, 1]
    test2 = MatrixMult(Func1, Func2)
    output00 = test2.forward(x)[0, 0]
    output01 = test2.forward(x)[0, 1]
    output10 = test2.forward(x)[1, 0]
    output11 = test2.forward(x)[1, 1]
    assert output00.allclose(matrix00, atol=0.0)
    assert output01.allclose(matrix01, atol=0.0)
    assert output10.allclose(matrix10, atol=0.0)
    assert output11.allclose(matrix11, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                 [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]],
                [[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]]],
               device=device, requires_grad=True)
    y = test.forward(x)
    y.sum().backward()
    # test __matmul__()
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    func1 = Composition((Identity((12,)), Unflatten((4, 3))))
    func2 = Composition((Identity((12,)), Unflatten((3, 4))))
    test4 = (func1 @ func2).to(device)
    assert isinstance(test4, MatrixMult)
    x = tensor([[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
                [13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]],
               device=device)
    desired = func1.forward(x[0]) @ func2.forward(x[0])
    assert test4.forward(x)[0].allclose(desired, atol=0.0)
    desired = func1.forward(x[1]) @ func2.forward(x[1])
    assert test4.forward(x)[1].allclose(desired, atol=0.0)
    Func1 = Affine(12, 3, *affine_init_torch(12, 3))
    Func2 = Composition((Identity((12,)), Unflatten((3, 4))))
    test5 = (func1 @ func2).to(device)
    desired = func1.forward(x[0]) @ func2.forward(x[0])
    assert test5.forward(x)[0].allclose(desired, atol=0.0)
    desired = func1.forward(x[1]) @ func2.forward(x[1])
    assert test5.forward(x)[1].allclose(desired, atol=0.0)


# test Polynomial
def test_Polynomial() -> None:
    """Test the ``Polynomial`` class.

    This method tests the ``Polynomial`` class.  In particular, this
    instantiates ``Polynomial`` and tests ``dims_in``, ``dims_out``,
    ``degrees``, ``coefficients``, ``forward()``, and
    ``torch.nn.Module`` backward differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the Polynomial
    dim_in = 2
    dim_out = 3
    degrees = (0, 1, 2)
    coefficients = polynomial_init_uniform(dim_in, dim_out, 1.0, degrees)
    try:
        Polynomial(dim_in, dim_out, degrees[:-1], coefficients)
        raise NotImplementedError("Test not implemented.")
    except ValueError as error:
        assert str(error) == ("The number of polynomial degrees and "
                              + "coefficients do not match.")
    try:
        coefficients = (coefficients[0], coefficients[1], coefficients[2][1:])
        Polynomial(dim_in, dim_out, degrees, coefficients)
        raise NotImplementedError("Test not implemented.")
    except ValueError as error:
        assert str(error) == ("The coefficient tensor is not the right "
                              + "shape.")
    coefficients = polynomial_init_uniform(dim_in, dim_out, 1.0, degrees)
    test = Polynomial(dim_in, dim_out, degrees, coefficients).to(device)
    # test dims_in
    assert test.dims_in == (2,)
    # test dims_out
    assert test.dims_out == (3,)
    # test degrees
    assert test.degrees == (0, 1, 2)
    # test coefficients
    assert test.coefficients[0].allclose(coefficients[0], atol=0.0)
    assert test.coefficients[1].allclose(coefficients[1], atol=0.0)
    assert test.coefficients[2].allclose(coefficients[2], atol=0.0)
    # test forward()
    x = tensor([[1., 2.],
                [3., 4.]], device=device)
    output1 = test.forward(x)[0]
    desired1 = (coefficients[0][0]
                + coefficients[1][0] * x[0, 0]
                + coefficients[1][1] * x[0, 1]
                + coefficients[2][0] * x[0, 1]**2
                + coefficients[2][1] * x[0, 0] * x[0, 1]
                + coefficients[2][2] * x[0, 0]**2)
    assert output1.allclose(desired1, atol=0.0)
    output2 = test.forward(x)[1]
    desired2 = (coefficients[0][0]
                + coefficients[1][0] * x[1, 0]
                + coefficients[1][1] * x[1, 1]
                + coefficients[2][0] * x[1, 1]**2
                + coefficients[2][1] * x[1, 0] * x[1, 1]
                + coefficients[2][2] * x[1, 0]**2)
    assert output2.allclose(desired2, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1., 2.],
                [3., 4.]], device=device, requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test Affine
def test_Affine() -> None:
    """Test the ``Affine`` class.

    This method tests the ``Affine`` class.  In particular, this
    instantiates ``Affine`` and tests ``dims_in``, ``dims_out``, ``degrees``,
    ``coefficients``, ``forward()``, and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the Affine with a bias
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    b_T, A_T = affine_init_torch(2, 3)
    test1 = Affine(2, 3, b_T, A_T).to(device)
    test2 = Affine(2, 3, None, A_T).to(device)
    test3 = Affine(2, 3, b_T, None).to(device)
    try:
        Affine(2, 3, None, None)
        raise NotImplementedError("Test not implemented.")
    except ValueError as error:
        assert str(error) == ("The bias and matrix cannot both be None.")
    # test dims_in
    assert test1.dims_in == (2,)
    assert test2.dims_in == (2,)
    assert test3.dims_in == (2,)
    # test dims_out
    assert test1.dims_out == (3,)
    assert test2.dims_out == (3,)
    assert test3.dims_out == (3,)
    # test degrees
    assert test1.degrees == (0, 1)
    assert test2.degrees == (1,)
    assert test3.degrees == (0,)
    # test coefficients
    assert test1.coefficients[0].allclose(b_T, atol=0.0)
    assert test1.coefficients[1].allclose(A_T, atol=0.0)
    assert test2.coefficients[0].allclose(A_T, atol=0.0)
    assert test3.coefficients[0].allclose(b_T, atol=0.0)
    # test forward()
    x = tensor([[1., 2.],
                [3., 4.]], device=device)
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    torchlinear = TorchLinear(2, 3).to(device)
    assert test1.forward(x).allclose(torchlinear(x), atol=0.0)
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    torchlinear = TorchLinear(2, 3, bias=False).to(device)
    assert test2.forward(x).allclose(torchlinear(x), atol=0.0)
    desired = stack((b_T[0], b_T[0]))
    assert test3.forward(x).allclose(desired, atol=0.0)


# test GELU
def test_GELU() -> None:
    """Test the ``GELU`` class.

    This method tests the ``GELU`` class. In particular, this
    instantiates ``GELU`` and tests ``dims_in``, ``dims_out``,
    ``forward()``, and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``.

    Args
        None

    Return
        None

    Raises
        None

    References
        None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the GELU class
    test = GELU((2,)).to(device)
    # test dim_in
    assert test.dims_in == (2,)
    # test dim_out
    assert test.dims_out == (2,)
    # test forward
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = x * 0.5 * (1.0 + erf(x / sqrt(tensor(2.0))))
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device,
               requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test Sigmoid
def test_Sigmoid() -> None:
    """Test the ``Sigmoid`` class.

    This method tests the ``Sigmoid`` class. In particular, this
    instantiates ``Sigmoid`` and tests ``dims_in``, ``dims_out``,
    ``forward()``, and ``torch.nn.Module`` backward
    differentiation with respect to ``forward()``.

    Args
        None

    Return
        None

    Raises
        None

    References
        None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the GELU class
    test = Sigmoid((2,)).to(device)
    # test dim_in
    assert test.dims_in == (2,)
    # test dim_out
    assert test.dims_out == (2,)
    # test forward
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device)
    desired = 1.0 / (1.0 + exp(-x))
    assert test.forward(x).allclose(desired, atol=0.0)
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], device=device,
               requires_grad=True)
    y = test.forward(x)
    y.sum().backward()


# test FullyConnMLP
def test_FullyConnMLP() -> None:
    """Test the ``FullConnMLP`` class.

    This method tests the ``FullConnMLP`` class. In particular, it
    instantiates a ``FullConnMLP`` object and tests ``TYPES``, ``dim_in``,
    ``dim_out``, ``layers``, ``forward()``, ``types()``, and
    ``torch.nn.Module`` backward differentiation with respect to ``forward()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the FullyConnMLP
    try:
        FullyConnMLP((2, 3), (10, 10), (3, 3), type='unknown')
        raise NotImplementedError("Test not implemented.")
    except ValueError as error:
        assert str(error) == ("The M.L.P. type is unknown.")
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    test1 = FullyConnMLP((2, 3), (10, 10), (3, 3),
                         type='gelu_torch').to(device)
    test2 = FullyConnMLP((2, 3), (10, 10), (3, 3),
                         type='sigmoid_torch').to(device)
    # test TYPES
    desired1 = {'gelu_torch': (GELU, affine_init_torch),
                'sigmoid_torch': (Sigmoid, affine_init_torch)}
    assert FullyConnMLP.TYPES == desired1
    # test dim_in
    assert test1.dims_in == (2, 3)
    assert test2.dims_in == (2, 3)
    # test dim_out
    assert test1.dims_out == (3, 3)
    assert test2.dims_out == (3, 3)
    # test layers
    assert len(test1.layers.funcs) == 7
    assert len(test2.layers.funcs) == 7
    assert isinstance(test1.layers.funcs[0], Flatten)
    assert isinstance(test2.layers.funcs[0], Flatten)
    assert test1.layers.funcs[0].dims_in == (2, 3)
    assert test2.layers.funcs[0].dims_in == (2, 3)
    assert test1.layers.funcs[0].dims_out == (6,)
    assert test2.layers.funcs[0].dims_out == (6,)
    assert isinstance(test1.layers.funcs[1], Affine)
    assert isinstance(test2.layers.funcs[1], Affine)
    assert test1.layers.funcs[1].dims_in == (6,)
    assert test2.layers.funcs[1].dims_in == (6,)
    assert test1.layers.funcs[1].dims_out == (10,)
    assert test2.layers.funcs[1].dims_out == (10,)
    assert isinstance(test1.layers.funcs[2], GELU)
    assert isinstance(test2.layers.funcs[2], Sigmoid)
    assert test1.layers.funcs[2].dims_in == (10,)
    assert test2.layers.funcs[2].dims_in == (10,)
    assert test1.layers.funcs[2].dims_out == (10,)
    assert test2.layers.funcs[2].dims_out == (10,)
    assert isinstance(test1.layers.funcs[3], Affine)
    assert isinstance(test2.layers.funcs[3], Affine)
    assert test1.layers.funcs[3].dims_in == (10,)
    assert test2.layers.funcs[3].dims_in == (10,)
    assert test1.layers.funcs[3].dims_out == (10,)
    assert test2.layers.funcs[3].dims_out == (10,)
    assert isinstance(test1.layers.funcs[4], GELU)
    assert isinstance(test2.layers.funcs[4], Sigmoid)
    assert test1.layers.funcs[4].dims_in == (10,)
    assert test2.layers.funcs[4].dims_in == (10,)
    assert test1.layers.funcs[4].dims_out == (10,)
    assert test2.layers.funcs[4].dims_out == (10,)
    assert isinstance(test1.layers.funcs[5], Affine)
    assert isinstance(test2.layers.funcs[5], Affine)
    assert test1.layers.funcs[5].dims_in == (10,)
    assert test2.layers.funcs[5].dims_in == (10,)
    assert test1.layers.funcs[5].dims_out == (9,)
    assert test2.layers.funcs[5].dims_out == (9,)
    assert isinstance(test1.layers.funcs[6], Unflatten)
    assert isinstance(test2.layers.funcs[6], Unflatten)
    assert test1.layers.funcs[6].dims_in == (9,)
    assert test2.layers.funcs[6].dims_in == (9,)
    assert test1.layers.funcs[6].dims_out == (3, 3)
    assert test2.layers.funcs[6].dims_out == (3, 3)
    # test forward()
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    linear1 = TorchLinear(6, 10).to(device)
    linear2 = TorchLinear(10, 10).to(device)
    linear3 = TorchLinear(10, 9, bias=False).to(device)
    x = tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0]]], device=device)
    desired2 = linear3(gelu(linear2(gelu(linear1(x.flatten(1))))))
    desired2 = desired2.unflatten(-1, (3, 3))
    assert test1.forward(x).allclose(desired2, atol=0.0)
    # test types()
    desired3 = ['gelu_torch', 'sigmoid_torch']
    assert FullyConnMLP.types() == desired3
    # test torch.nn.Module backward differentiation for forward()
    x = tensor([[[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0],
                 [10.0, 11.0, 12.0]]], device=device, requires_grad=True)
    y = test1(x)
    y.sum().backward()
