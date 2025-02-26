"""Define the discrete dynamical system representation.

This module defines the discrete dynamical system representation.
"""


# import built-in python-package code
from abc import ABC, abstractmethod
# import external python-package code
from torch import Tensor
from torch.nn import Module
# import internal python-package code
# None


# export public code
__all__ = ["DiscreteSystem"]


# define a discrete dynamical system
class DiscreteSystem(ABC, Module):
    """Represent a discrete dynamical system.

    This class represents a discrete dynamical system.

    | **Abstract Attributes**
    |   ``num_states`` (``int``): the number of states

    | **Class Attributes**
    |   None

    | **Attributes**
    |   None

    | **Abstract Methods**
    |   ``map()``: return the mapping of a given input state

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclasses

    | **References**
    |   None
    """

    @property
    @abstractmethod
    def num_states(self) -> int:
        ...

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        """Return the mapping of a given input state.

        This method returns the mapping of a given input state.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (self.num_states,)``

        | **Return**
        |   ``Tensor``: the mapping of the input state with shape
                ``(...,) + (self.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    def __init__(self) -> None:
        """Initialize the superclasses.

        This method initializes the superclasses ``ABC`` and ``Module``.

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
