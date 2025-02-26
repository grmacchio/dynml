"""Test the dynml.dyn.discrete.system module.

This module tests the dynml.dyn.discrete.system module.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import float64, ones, set_default_dtype, Tensor, zeros
from torch.cuda import is_available
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.discrete.system import DiscreteSystem


# test the DiscreteSystem class
class DiscreteSystemExample(DiscreteSystem):
    """Represent a subclass of ``DiscreteSystem``.

    This class represents a subclass of ``DiscreteSystem``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``num_states`` (``int``): the number four

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``map()``: the input plus one
    |   ``__init__()``: initialize the superclass and attributes

    | **References**
    |   None
    """

    @property
    def num_states(self) -> int:
        return self._num_states

    def map(self, x: Tensor) -> Tensor:
        """Return the input plus one.

        This method returns the input plus one.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape ``(...,) + (4,)``

        | **Return**
        |   ``Tensor``: the input plus one with shape ``(...,) + (4,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return x + self.ones

    def __init__(self, num_states: int) -> None:
        """Initialize the superclass and the attributes.

        This method initializes the superclass and the attributes.

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
        # initialize the attributes
        self._num_states = num_states
        self.ones = Parameter(ones((num_states,)))


def test_DiscreteSystem() -> None:
    """Test the ``DiscreteSystem`` class.

    This method tests the ``DiscreteSystem`` class. In particular, it
    instantiates ``DiscreteSystemExample`` and tests ``num_states``, and
    ``map()``.

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
    # initialize DiscreteSystemExample
    num_states = 4
    test = DiscreteSystemExample(num_states).to(device)
    # test num_states
    assert test.num_states == num_states
    # test map
    x = zeros((2, 4, num_states), device=device)
    assert test.map(x).allclose(x + 1.0, atol=0.0)
