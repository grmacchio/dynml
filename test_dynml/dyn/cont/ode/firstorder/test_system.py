"""Test the ``dynml.dyn.cont.ode.firstorder.system`` module.

This module tests the ``dynml.dyn.cont.ode.firstorder.system`` module.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import float64, set_default_dtype, tensor, Tensor
from torch.nn import Parameter
from torch.cuda import is_available
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import FirstOrderSystem
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# test FirstOrderSystem
class FirstOrderSystemExample(FirstOrderSystem):
    """Represent a subclass of ``FirstOrderSystem``.

    This class represents a subclass of ``FirstOrderSystem``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): ``R`` for real numbers
    |   ``num_states`` (``int``): the number two

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``rhs()``: the identity times two
    |   ``__init__()``: initialize the superclass

    | **References**
    |   None
    """

    @property
    def field(self) -> str:
        return 'R'

    @property
    def num_states(self) -> int:
        return 2

    def rhs(self, x: Tensor) -> Tensor:
        """Return the input times two.

        This method returns the input times two.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape
                ``(...,) + (2,)``

        | **Return**
        |   ``Tensor``: the input times two with shape
                ``(...,) + (2,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the input times two
        return 2 * x

    def __init__(self) -> None:
        """Initialize the superclass.

        This method initializes the superclass.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__()


def test_FirstOrderSystem() -> None:
    """Test the ``FirstOrderSystem`` class.

    This method tests the ``FirstOrderSystem`` class. In particular, this
    method instantiates a subclass of ``FirstOrderSystem`` and tests,
    ``field``, ``num_states``, and ``rhs()``.

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
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate the subclass of FirstOrderSystem
    test = FirstOrderSystemExample().to(device)
    # test field
    assert test.field == 'R'
    # test num_states
    assert test.num_states == 2
    # test rhs
    x = tensor([[1., 2.], [3., 4.]], device=device)
    desired = 2 * x
    assert test.rhs(x).allclose(desired, atol=0.0)
    assert test.rhs(x).shape == (2, 2)


# test SemiLinearFirstOrderSystem
class SemiLinearFirstOrderSystemExample(SemiLinearFirstOrderSystem):
    """Represent a subclass of ``SemiLinearFirstOrderSystem``.

    This class represents a subclass of ``SemiLinearFirstOrderSystem``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``A`` (``Parameter``): the matrix ``torch.tensor([[2, 1], [1, 2]])``
    |   ``field`` (``str``): ``R`` for real numbers
    |   ``num_states`` (``int``): the number two

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``nonlinear()``: the identity times three
    |   ``rhs()``: the right-hand-side evaluated at some state
    |   ``__init__()``: initialize the superclass and model parameters

    | **References**
    |   None
    """

    @property
    def A(self) -> Tensor:
        return self._A

    @property
    def field(self) -> str:
        return 'R'

    @property
    def num_states(self) -> int:
        return 2

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return the input times three.

        This method returns the input times three.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape ``(...,) + (2,)``

        | **Return**
        |   ``Tensor``: the input times three with shape ``(...,) + (2,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the input times three
        return 3 * x

    def __init__(self) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__()
        # initialize the model parameters
        self._A = Parameter(tensor([[2., 1.], [1., 2.]]), requires_grad=False)


def test_SemiLinearFirstOrderSystem() -> None:
    """Test the ``SemiLinearFirstOrderSystem`` class.

    This method tests the ``SemiLinearFirstOrderSystem`` class. In particular,
    this method instantiates a subclass of ``SemiLinearFirstOrderSystem`` and
    tests ``A``, ``field``, ``num_states``, and ``nonlinear()``.

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
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # instantiate SemiLinearFirstOrderSystemExample
    test = SemiLinearFirstOrderSystemExample().to(device)
    # test L
    desired = tensor([[2., 1.], [1., 2.]], device=device)
    assert test.A.allclose(desired, atol=0.0)
    assert test.A.shape == (2, 2)
    # test field
    assert test.field == 'R'
    # test num_states
    assert test.num_states == 2
    # test nonlinear
    x = tensor([[1., 2.], [3., 4.]], device=device)
    desired = 3 * x
    assert test.nonlinear(x).allclose(desired, atol=0.0)
    assert test.nonlinear(x).shape == (2, 2)
    # test rhs
    A = tensor([[2., 1.], [1., 2.]], device=device)
    desired = x @ A.T + 3 * x
    assert test.rhs(x).allclose(desired, atol=0.0)
    assert test.rhs(x).shape == (2, 2)
