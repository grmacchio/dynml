"""Contain all code related to a system with a paraboloid slow manifold.

This module contains all code related to a system with a paraboloid slow
manifold.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import rand, randn, sqrt, tensor, Tensor, zeros_like
from torch.linalg import norm
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# export public code
__all__ = ['Paraboloid']


# argument parser choice dictionaries
class Paraboloid(SemiLinearFirstOrderSystem):
    """Represent a system with a paraboloid slow manifold.

    This class represents a system with a paraboloid slow manifold. In
    particular, this is a three-state cylinder wake model described in [1, 2].
    The model is given by the following system of real-valued ordinary
    differential equations:

    .. math::
        \\begin{align*}
            \\dot{x} = \\begin{bmatrix}
                \\dot{x}_1 \\\\
                \\dot{x}_2 \\\\
                \\dot{x}_3
            \\end{bmatrix}
            = \\begin{bmatrix}
                \\mu & -\\omega & 0 \\\\
                \\omega & \\mu & 0 \\\\
                0 & 0 & -\\lambda
            \\end{bmatrix}
            \\begin{bmatrix}
                x_1 \\\\
                x_2 \\\\
                x_3
            \\end{bmatrix} +
            \\begin{bmatrix}
                \\alpha x_1 x_3 \\\\
                \\alpha x_2 x_3 \\\\
                \\lambda (x_1^2 + x_2^2)
            \\end{bmatrix}
            = A x + F(x) = f(x),
        \\end{align*}

    where :math:`\\mu`, :math:`\\omega`, :math:`\\alpha`, and :math:`\\lambda`
    are real-valued parameters.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): ``R`` for real numbers
    |   ``dims_state`` (``int``): the state dimensions
    |   ``A`` (``Tensor``): the matrix :math:`A`
    |   ``mu`` (``float``): the value of the parameter :math:`\\mu` with
            default value of ``0.1``
    |   ``omega`` (``float``): the value of the parameter :math:`\\omega` with
            default value of ``1.0``
    |   ``alpha`` (``float``): the value of the parameter :math:`\\alpha` with
            default value of ``-0.1``
    |   ``lam`` (``float``): the value of the parameter :math:`\\lambda` with
            default value of ``10.0``

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``nonlinear()``: return :math:`F(x)`
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``taylor_approx_slow_manifold()``: return the 4th-ord. approx. slow
            manifold :math:`x_3` coordinate
    |   ``gen_ic()``: return an initial condition where
            :math:`\\|x\\|_2 \\sim U[[0, 1)]`
    |   ``rhs()``: return :math:`f(x)`

    | **References**
    |   [1] Noack, Bernd R., et al. "A hierarchy of low-dimensional models for
            the transient and post-transient cylinder wake." Journal of Fluid
            Mechanics 497 (2003): 335-363, p. 337.
    |   [2] Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. "Deep
            learning for universal linear embeddings of nonlinear dynamics."
            Nature communications 9.1 (2018): 4950, p. 5.
    """

    @property
    def field(self) -> str:
        return 'R'

    @property
    def dims_state(self) -> Tuple[int, ...]:
        return (3,)

    @property
    def A(self) -> Tensor:
        return self._A

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(x)`.

        This method returns :math:`F(x)`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape ``(...,) + (3,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape ``(...,) + (3,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Noack, Bernd R., et al. "A hierarchy of low-dimensional models
                for the transient and post-transient cylinder wake." Journal of
                Fluid Mechanics 497 (2003): 335-363, p. 337.
        |   [2] Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. "Deep
                learning for universal linear embeddings of nonlinear
                dynamics." Nature communications 9.1 (2018): 4950, p. 5.
        """
        # compute the nonlinear term
        out = zeros_like(x)
        out[..., 0] = self.alpha * x[..., 0] * x[..., 2]
        out[..., 1] = self.alpha * x[..., 1] * x[..., 2]
        out[..., 2] = self.lam * (x[..., 0]**2 + x[..., 1]**2)
        return out

    def __init__(self, mu: float = 0.1, omega: float = 1.0,
                 alpha: float = -0.1, lam: float = 10.0):
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``mu`` (``float``): the value of the parameter :math:`\\mu` with
                default value of ``0.1``
        |   ``omega`` (``float``): the value of the parameter :math:`\\omega`
                with default value of ``1.0``
        |   ``alpha`` (``float``): the value of the parameter :math:`\\alpha`
                with default value of ``-0.1``
        |   ``lam`` (``float``): the value of the parameter :math:`\\lambda`
                with default value of ``10.0``

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   [1] Noack, Bernd R., et al. "A hierarchy of low-dimensional models
                for the transient and post-transient cylinder wake." Journal of
                Fluid Mechanics 497 (2003): 335-363, p. 337.
        |   [2] Lusch, Bethany, J. Nathan Kutz, and Steven L. Brunton. "Deep
                learning for universal linear embeddings of nonlinear
                dynamics." Nature communications 9.1 (2018): 4950, p. 5.
        """
        # initialize the superclass and model parameters
        super().__init__()
        self.mu = mu
        self.omega = omega
        self.alpha = alpha
        self.lam = lam
        self._A = Parameter(tensor([[self.mu, -self.omega, 0.0],
                                    [self.omega, self.mu, 0.0],
                                    [0.0, 0.0, -self.lam]]),
                            requires_grad=False)

    def taylor_approx_slow_manifold(self, v: Tensor) -> Tensor:
        """Return the 4th-ord. approx. slow manifold :math:`x_3` coordinate.

        This method returns the 4th-ord. approx. slow manifold :math:`x_3`
        coordinate given an :math:`x_1`-:math:`x_2` plane / slow subspace
        coordinate. Let :math:`h` be the graph function from the
        :math:`x_1`-:math:`x_2` plane to the slow manifold :math:`x_3`
        coordinate.

        | **Args**
        |   ``v`` (``Tensor``): the :math:`x_1`-:math:`x_2` plane coordinate,
                :math:`v`, with shape ``(...,) + (2,)``

        | **Returns**
        |   ``Tensor``: the slow manifold :math:`x_3`-point :math:`h(v)` with
                shape ``(...,) + (1,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Otto, Samuel E., Gregory R. Macchio, and Clarence W. Rowley.
                "Learning nonlinear projections for reduced-order modeling of
                dynamical systems using constrained autoencoders." Chaos: An
                Interdisciplinary Journal of Nonlinear Science 33.11 (2023).
        """
        # compute the radius
        r = sqrt(v[..., 0]**2 + v[..., 1]**2)
        # computer the 0th-4th order terms: h0, h1, h2, h3, h4
        h0 = r**2
        h1 = - 2 * r**2 * (self.alpha * r**2 + self.mu)
        h2 = 4 * r**2 * (3 * self.alpha**2 * r**4
                         + 4 * self.alpha * r**2 * self.mu
                         + self.mu**2)
        h3 = -8 * r**2 * (14 * self.alpha**3 * r**6
                          + 24 * self.alpha**2 * r**4 * self.mu
                          + 11 * self.alpha * r**2 * self.mu**2
                          + self.mu**3)
        h4 = 16 * r**2 * (85 * self.alpha**4 * r**8
                          + 180 * self.alpha**3 * r**6 * self.mu
                          + 120 * self.alpha**2 * r**4 * self.mu**2
                          + 26 * self.alpha * r**2 * self.mu**3
                          + self.mu**4)
        # return the 4th-order approx. slow manifold x3 coordinate
        return (h0
                + h1 * (1 / self.lam)
                + h2 * (1 / self.lam)**2
                + h3 * (1 / self.lam)**3
                + h4 * (1 / self.lam)**4)

    def gen_ic(self) -> Tensor:
        """Return an initial condition where :math:`\\|x\\|_2 \\sim U[[0, 2)]`.

        This method returns an initial condition where :math:`\\|x\\|_2 \\sim
        U[[0, 1)]`. In particular, an initial condition is sampled in the
        following way: First, :math:`u \\sim U[\\mathcal{S}^{2}]`. Second,
        :math:`r\\sim U[[0, 1)]`. Finally, the sample :math:`ru` is returned.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(3,)``

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
