"""Define some first-order O.D.E. dynamical-system representations.

This module defines some first-order O.D.E. dynamical-system representations.
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
__all__ = ["FirstOrderSystem",
           "SemiLinearFirstOrderSystem"]


# define some first-order O.D.E. dynamical system representations
class FirstOrderSystem(ABC, Module):
    """Represent a first-order O.D.E. system.

    This abstract class represents a first-order O.D.E. system.

    | **Abstract Attributes**
    |   ``field`` (``str``): the field the dynamical system is defined over
    |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions

    | **Class Attributes**
    |   None

    | **Attributes**
    |   None

    | **Abstract Methods**
    |   ``rhs()``: return the right-hand-side evaluated at some state

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
    def rhs(self, x: Tensor) -> Tensor:
        """Return the right-hand-side evaluated at some state.

        This method returns the right-hand-side evaluated at some state.

        | **Args**
        |   ``x`` (``Tensor``): a state with shape
                ``(...,) + self.dims_state``

        | **Return**
        |   ``Tensor``: the right-hand-side with shape
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
        # call the superclass constructors
        super().__init__()


class SemiLinearFirstOrderSystem(FirstOrderSystem):
    """Represent a semi-linear first-order O.D.E. system.

    This abstract class represents a semi-linear first-order O.D.E. system.
    A semi-linear first-order O.D.E. system is a first-order O.D.E system whose
    right-hand-side is the sum of a linear and a nonlinear function.

    | **Abstract Attributes**
    |   ``A`` (``Tensor``): the matrix representation of the linear function
    |   ``field`` (``str``): the field the dynamical system is defined over
    |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions

    | **Class Attributes**
    |   None

    | **Attributes**
    |   None

    | **Abstract Methods**
    |   ``nonlinear()``: return the nonlinear function evaluated at some state

    | **Class Methods**
    |   None

    | **Methods**
    |   ``rhs()``: the right-hand-side evaluated at some state.
    |   ``__init__()``: initialize the superclass

    | **References**
    |   None
    """

    @property
    @abstractmethod
    def A(self) -> Tensor:
        ...

    @abstractmethod
    def nonlinear(self, x: Tensor) -> Tensor:
        """Return the nonlinear function evaluated at some state.

        This method returns the nonlinear function evaluated at some state.

        | **Args**
        |   ``x`` (``Tensor``): a state with shape
                ``(...,) + self.dims_state``

        | **Return**
        |   ``Tensor``: the nonlinear function with shape
                ``(...,) + self.dims_state``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    def rhs(self, x: Tensor) -> Tensor:
        """Return the right-hand-side evaluated at some state.

        This method returns the right-hand-side evaluated at some state.

        | **Args**
        |   ``x`` (``Tensor``): a state with shape
                ``(...,) + self.dims_state``

        | **Return**
        |   ``Tensor``: the right-hand-side evaluated at the state with shape
                ``(...,) + self.dims_state``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the linear function plus the nonlinear function
        return x @ self.A.T + self.nonlinear(x)

    def __init__(self) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``FirstOrderSystem``.

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
