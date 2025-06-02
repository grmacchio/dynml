"""Contain all code related to a system with a parabolic slow manifold.

This module contains all code related to a system with a parabolic slow
manifold.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import rand, randn, tensor, Tensor, zeros_like
from torch.nn import Parameter
from torch.linalg import norm
# import internal python-package code
from dynml.dyn.cont.ode.firstorder import SemiLinearFirstOrderSystem


# export public code
__all__ = ['Parabola']


# define the dynamic system
class Parabola(SemiLinearFirstOrderSystem):
    """Represent a system with a parabolic slow manifold.

    This class represents a system with a parabolic slow manifold. In
    particular, the toy nonlinear system is defined by the following system of
    first-order ordinary differential equations:

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                \\dot{x}_1 \\\\
                \\dot{x}_2
            \\end{bmatrix} &=
            \\begin{bmatrix}
                -\\lambda_1 x_1 (x_1^2 - 1) \\\\
                -\\lambda_2 (x_2 - x_1^2)
            \\end{bmatrix} = f(x) = Ax + F(x).
        \\end{align*}

    where :math:`0 < \\lambda_1 \\ll \\lambda_2`.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): ``R`` for real numbers
    |   ``lambda1`` (``float``): the parameter :math:`\\lambda_1` with default
    |       value :math:`1.0`
    |   ``lambda2`` (``float``): the parameter :math:`\\lambda_2` with default
    |       value :math:`100.0`
    |   ``A`` (``Tensor``): the matrix :math:`A`
    |   ``dims_state`` (``int``): the state dimensions

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``nonlinear()``: return :math:`F(x)`
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``rhs()``: return :math:`f(x)`
    |   ``gen_ic()``: return an I.C. where :math:`\\|x\\|_2 \\sim U[[0, 1)]`

    | **References**
    |   None
    """

    @property
    def field(self) -> str:
        return 'R'

    @property
    def A(self) -> Tensor:
        return self._A

    @property
    def dims_state(self) -> Tuple[int, ...]:
        return (2,)

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(x)`.

        This method returns :math:`F(x)`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape ``(...,) +(2,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape ``(...,) +(2,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # compute the nonlinear term
        output = zeros_like(x)
        output[..., 0] = -self.lambda1 * x[..., 0]**3
        output[..., 1] = self.lambda2 * x[..., 0]**2
        return output

    def __init__(self, lambda1: float = 1.0, lambda2: float = 10.0) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``lambda1`` (``float``): the parameter :math:`\\lambda_1` with
        |       default value :math:`1.0`
        |   ``lambda2`` (``float``): the parameter :math:`\\lambda_2` with
                default value :math:`10.0`

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__()
        # set the model parameters
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self._A = Parameter(tensor([[lambda1, 0.0], [0.0, -lambda2]]),
                            requires_grad=False)
        self._num_states = 2

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|x\\|_2 \\sim U[[0, 1)]`.

        This method returns an I.C. where :math:`\\|x\\|_2 \\sim
        U[[0, 1)]`. In particular, an initial condition is sampled in the
        following way: First, :math:`u \\sim U[\\mathcal{S}^{n-1}]`.
        Second, :math:`r\\sim U[[0, 1)]`. Finally, the sample :math:`ru` is
        returned.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape ``(2,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the initial condition where ||x||_2 ~ U[[0, 1)]
        gaussian = randn((self.dims_state[-1],),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = rand((1,), device=next(self.parameters()).device.type)
        return radius * direction
