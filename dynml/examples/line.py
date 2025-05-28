"""Contain all code related to a system with a linear slow manifold.

This module contains all code related to a system with a linear slow manifold.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import exp, rand, randn, tensor, Tensor, zeros_like
from torch.nn import Parameter
from torch.linalg import norm
# import internal python-package code
from dynml.dyn.cont.ode.firstorder import SemiLinearFirstOrderSystem


# export public code
__all__ = ['Line']


# define the dynamic system
class Line(SemiLinearFirstOrderSystem):
    """Represent a system with a linear slow manifold.

    This class represents a system with a linear slow manifold. In
    particular, the toy nonlinear system is defined by the following system of
    first-order ordinary differential equations:

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                \\dot{x}_1 \\\\
                \\dot{x}_2
            \\end{bmatrix} &=
            \\begin{bmatrix}
                -\\lambda_1 x_1 + C x_1 x_2 \\\\
                -\\lambda_2 x_2
            \\end{bmatrix} = f(x) = Ax + F(x).
        \\end{align*}

    where :math:`0 < \\lambda_1 \\ll \\lambda_2 < C`. The origin is
    the only fixed point, and the linearization about the origin is diagonal.
    Furthermore, the :math:`x_1`-axis and :math:`x_2`-axis are each invariant
    subspaces. Nevertheless, the system has an interesting dynamical feature:
    if :math:`C` is large, then there can be large transient growth in the
    :math:`x_1`-coordinate, if the initial values of :math:`x_1` and
    :math:`x_2` are not too small. To understand this feature, we consider the
    linearizing coordinate transformation

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                x_1 \\\\
                x_2
            \\end{bmatrix} = h(z) =
            \\begin{bmatrix}
                z_1 \\exp(-C z_2 / \\lambda_2) \\\\
                z_2
            \\end{bmatrix}
            \\qquad
            \\begin{bmatrix}
                z_1 \\\\
                z_2
            \\end{bmatrix} = h^{-1}(x) =
            \\begin{bmatrix}
                x_1 \\exp(C x_2 / \\lambda_2) \\\\
                x_2
            \\end{bmatrix},
        \\end{align*}

    which transforms the nonlinear system into the following linear system:

    .. math::
        \\begin{align*}
            \\begin{bmatrix}
                \\dot{z}_1 \\\\
                \\dot{z}_2
            \\end{bmatrix}
            &= Dh^{-1}(x) f(x)
            \\\\
            &= \\begin{bmatrix}
                \\exp(C x_2 / \\lambda_2) (-\\lambda_1 x_1 + C x_1 x_2) +
                \\frac{C x_1}{\\lambda_2} \\exp(C x_2 / \\lambda_2)
                (-\\lambda_2 x_2) \\\\
                0 \\cdot (-\\lambda_1 x_1 + C x_1 x_2)
                + 1 \\cdot (-\\lambda_2 x_2)
            \\end{bmatrix}
            \\\\
            &= \\begin{bmatrix}
            \\exp(C x_2 / \\lambda_2) (-\\lambda_1 x_1) \\\\
            -\\lambda_2 x_2
            \\end{bmatrix}
            = \\begin{bmatrix}
                -\\lambda_1 z_1 \\\\
                -\\lambda_2 z_2
            \\end{bmatrix}.
        \\end{align*}

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
    |   ``C`` (``float``): the parameter :math:`C` with default value
    |       :math:`300.0`
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
    |   ``h()``: return the transformation :math:`h(z)`
    |   ``h_inv()``: return the inverse transformation :math:`h^{-1}(x)`

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
        output[..., 0] = self.C * x[..., 0] * x[..., 1]
        return output

    def __init__(self, lambda1: float = 1.0, lambda2: float = 100.0,
                 C: float = 300.0) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``lambda1`` (``float``): the parameter :math:`\\lambda_1` with
        |       default value :math:`1.0`
        |   ``lambda2`` (``float``): the parameter :math:`\\lambda_2` with
                default value :math:`10.0`
        |   ``C`` (``float``): the parameter :math:`C` with default value
                :math:`30.0`

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
        self.C = C
        self._A = Parameter(tensor([[-lambda1, 0.0], [0.0, -lambda2]]),
                            requires_grad=False)

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
        gaussian = randn((self.num_states,),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = rand((1,), device=next(self.parameters()).device.type)
        return radius * direction

    def h(self, z: Tensor) -> Tensor:
        """Return the transformation :math:`h(z)`.

        This method returns the transformation :math:`h(z)`.

        | **Args**
        |   ``z`` (``Tensor``): the state with shape ``(...,) +(2,)``

        | **Returns**
        |   ``Tensor``: the transformation with shape ``(...,) +(2,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the transformation h(z)
        output = zeros_like(z)
        output[..., 0] = z[..., 0] * exp(-self.C * z[..., 1] / self.lambda2)
        output[..., 1] = z[..., 1]
        return output

    def h_inv(self, x: Tensor) -> Tensor:
        """Return the inverse transformation :math:`h^{-1}(x)`.

        This method returns the inverse transformation :math:`h^{-1}(x)`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape ``(...,) +(2,)``

        | **Returns**
        |   ``Tensor``: the inverse transformation with shape ``(...,) +(2,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the inverse transformation h^{-1}(x)
        output = zeros_like(x)
        output[..., 0] = x[..., 0] * exp(self.C * x[..., 1] / self.lambda2)
        output[..., 1] = x[..., 1]
        return output
