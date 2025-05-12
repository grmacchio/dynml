"""Define the discrete dynamical system representation.

This module defines the discrete dynamical system representation.
"""


# import built-in python-package code
from abc import ABC, abstractmethod
from typing import Tuple
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
    |   ``field`` (``str``): the field the dynamical system is defined over
    |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions

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
    def field(self) -> str:
        """Return the field the dynamical system is defined over.

        This method returns the field the dynamical system is defined over. The
        allowable field are ``R`` and ``C`` for real and complex numbers,
        respectively.

        | **Args**
        |   None

        | **Return**
        |   ``str``: the field the dynamical system is defined over

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    @property
    @abstractmethod
    def dims_state(self) -> Tuple[int, ...]:
        ...

    @abstractmethod
    def map(self, x: Tensor) -> Tensor:
        """Return the mapping of a given input state.

        This method returns the mapping of a given input state.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + self.dims_state``

        | **Return**
        |   ``Tensor``: the mapping of the input state with shape
                ``(...,) + self.dims_state``

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
