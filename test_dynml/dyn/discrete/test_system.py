"""Test the dynml.dyn.discrete.system module.

This module tests the dynml.dyn.discrete.system module.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import ones, Tensor, zeros
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.discrete.system import DiscreteSystem
from dynml.utils.config import config


# test the DiscreteSystem class
class DiscreteSystemExample(DiscreteSystem):
    """Represent a subclass of ``DiscreteSystem``.

    This class represents a subclass of ``DiscreteSystem``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``dims_state`` (``int``): the state dimensions

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
    def dims_state(self) -> Tuple[int, ...]:
        return self._dims_state

    def map(self, x: Tensor) -> Tensor:
        """Return the input plus one.

        This method returns the input plus one.

        | **Args**
        |   ``x`` (``Tensor``): the input with shape ``(...,) +
                self.dims_state``

        | **Return**
        |   ``Tensor``: the input plus one with shape ``(...,) +
                self.dims_state``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return self._map(x)

    def __init__(self, dims_state: Tuple[int, ...], params: bool) -> None:
        """Initialize the superclass and the attributes.

        This method initializes the superclass and the attributes.

        | **Args**
        |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions
        |   ``params`` (``bool``): the parameters boolean

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
        self._dims_state = dims_state
        if params:
            self._ones_param: Parameter = Parameter(ones(self._dims_state))
            self._map = lambda x: x + self._ones_param
        else:
            self._ones_tensor: Tensor = ones(self._dims_state)
            self._map = lambda x: x + self._ones_tensor


def test_DiscreteSystem() -> None:
    """Test the ``DiscreteSystem`` class.

    This method tests the ``DiscreteSystem`` class. In particular, it
    instantiates ``DiscreteSystemExample`` and tests ``dims_state``,
    and ``map()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # configure the test
    device = config(64, 0)
    # initialize DiscreteSystemExample
    dims_state = (4, 3)
    test = DiscreteSystemExample(dims_state, False).to(device)
    # test num_states
    assert test.dims_state == dims_state
    # test map
    x = zeros((2, 4) + dims_state, device=device)
    assert test.map(x).allclose(x + 1.0, atol=0.0)
