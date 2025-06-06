"""Contain all code related to the complex Ginzburg-Landau dynamical system.

This module contains all code related to the complex Ginzburg-Landau dynamical
system.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import arange, cat, conj, pi, rand, randn
from torch import Tensor, zeros
from torch.fft import ifft, fft
from torch.linalg import norm
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# export public code
__all__ = ['CGLE']


class CGLE(SemiLinearFirstOrderSystem):
    """Represent the discretized complex Ginzburg-Landau dynamical system with
    periodic boundary conditions.

    This class represents the discretized complex Ginzburg-Landau dynamical
    system with periodic boundary conditions on :math:`[0, L]`. The complex
    Ginzburg-Landau equation is a partial differential equation of the form
    described in [1, 2]:

    .. math::
        \\frac{\\partial u}{\\partial t}
        = \\left(\\alpha + \\beta
        \\frac{\\partial^2}{\\partial x^2}\\right)u
        + \\gamma \\left|u\\right|^2 u

    where :math:`u(t, x), \\alpha, \\beta, \\gamma \\in \\mathbb{C}`,
    :math:`t \\in \\mathbb{R}`, and :math:`x \\in [0, L]`. In order to obtain a
    first-order O.D.E. system, we orthogonally project this P.D.E. onto the
    space of complex Fourier modes where solutions :math:`u` take the form,

    .. math::
        \\begin{align*}
            u = \\sum_{k=-K}^{K} e^{\\frac{2\\pi i k}{L} \\:\\cdot_x} \\:
            U_k(\\:\\cdot_{t}).
        \\end{align*}

    By substituting this expression into the P.D.E. and projecting using the
    the :math:`1 / L` forward-normalized Fourier transform
    :math:`\\mathcal{F}_{k}` on :math:`[0, L]`, we can write
    the complex Ginzburg-Landau equation as a system of complex-valued ordinary
    differential equations:

    .. math::
        \\begin{align*}
            \\dot{U}_{k} &= \\left(\\alpha - \\beta
            \\left(\\frac{2\\pi k}{L} \\right)^2\\right) U_k
            + \\mathcal{F}_{k} \\left( \\gamma
            |u|^2 u \\right)
        \\end{align*}

    for :math:`k \\in [-K : K]`. The nonlinear term is approximated using
    the :math:`1 / (2K + 1)` forward-normalized discrete Fourier transform
    :math:`\\mathcal{D}_k` on :math:`[-K : K]` and applying the nonlinearity to
    the solution's values at the collocation points:

    .. math::
        \\begin{align*}
            \\mathcal{F}_{k}\\left(\\gamma |u|^2 u\\right)
            \\approx
            \\mathcal{D}_{k}\\left[
                \\left(
                \\gamma \\cdot
                |\\mathcal{D}_{n}^{-1} \\left[
                    (U_{m})_{m\\:}
                \\right]|^2
                \\cdot
                \\mathcal{D}_{n}^{-1} \\left[
                    (U_{m})_{m\\:}
                \\right]
            \\right)_{n\\:}
            \\right]
        \\end{align*}

    where :math:`m \\in [-K : K]` is the frequency index and
    :math:`n \\in [0 : 2K]` is the collocation point index [3]. When using the
    discrete fourier transform we account for aliasing by using the
    :math:`3/2` rule [3]. The final system takes the form,

    .. math::
        \\begin{align*}
            \\dot{\\vec{x}} = A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    where

    .. math::
        \\begin{align*}
            \\vec{x} &= (\\text{Re} \\: U_{0}, \\ldots, \\text{Re} \\: U_{K},
            \\text{Re} \\: U_{-K}, \\ldots,  \\text{Re} \\: U_{-1},
            \\text{Im} \\: U_{0}, \\ldots, \\text{Im} \\: U_{K},
            \\text{Im} \\: U_{-K} \\ldots \\text{Im} \\: U_{-1})
        \\end{align*}

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): ``R`` for complex numbers
    |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions
    |   ``A`` (``Tensor``): the matrix :math:`A`
    |   ``K`` (``int``): the number of Fourier modes :math:`K` with default
            value of ``256``
    |   ``N`` (``int``): the number of collocation points :math:`N = 2K + 1`
    |   ``alpha`` (``float``): the value of the parameter
            :math:`\\alpha = \\alpha_r + i \\alpha_i` with default value
            of ``1.0 + 0.0j``
    |   ``L`` (``float``): the value of the parameter :math:`L` with default
            value of ``10.0``
    |   ``beta`` (``complex``): the value of the parameter
            :math:`\\beta` with default value of ``1.0 + 1.0j``
    |   ``gamma`` (``complex``): the value of the parameter
            :math:`\\gamma` with default value of ``-1.0 - 2.0j``

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``nonlinear()``: return :math:`F(\\vec{x})`
    |   ``rhs()``: return :math:`f(\\vec{x})`
    |   ``gen_ic()``: return an I.C. where
            :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`

    | **References**
    |   [1] 1988 - Doering - Low-Dimensional Behavior In the Complex
            Ginzburg-Landau Equation
    |   [2] 1993 - Doelman - Exponential Convergence of The Galerkin
            Approximation for the Ginzburg-Landau Equation
    |   [3] Peyret, Roger. Spectral methods for incompressible viscous flow.
            Vol. 148. New York: Springer, 2002, ch. 2.
    """

    @property
    def field(self) -> str:
        return 'R'

    @property
    def A(self) -> Tensor:
        return self._A

    @property
    def dims_state(self) -> Tuple[int, ...]:
        return (2 * (2 * self.K + 1),)

    def __init__(self, K: int = 256, L: float = 10.0,
                 alpha_r: float = 1.0, alpha_i: float = 0.0,
                 beta_r: float = 1.0, beta_i: float = 1.0,
                 gamma_r: float = -1.0, gamma_i: float = -2.0) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``K`` (``int``): the number of Fourier modes :math:`K` with a
                default value of ``256``
        |   ``L`` (``float``): the length of the domain :math:`L` with a
                default value of ``10.0``
        |   ``alpha_r`` (``float``): the real part of the parameter
                :math:`\\alpha` with a default value of ``1.0``
        |   ``alpha_i`` (``float``): the imaginary part of the parameter
                :math:`\\alpha` with a default value of ``0.0``
        |   ``beta_r`` (``float``): the real part of the parameter
                :math:`\\beta` with a default value of ``1.0``
        |   ``beta_i`` (``float``): the imaginary part of the parameter
                :math:`\\beta` with a default value of ``1.0``
        |   ``gamma_r`` (``float``): the real part of the parameter
                :math:`\\gamma` with a default value of ``-1.0``
        |   ``gamma_i`` (``float``): the imaginary part of the parameter
                :math:`\\gamma` with a default value of ``-2.0``

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   [1] 1988 - Doering - Low-Dimensional Behavior In the Complex
                Ginzburg-Landau Equation
        |   [2] 1993 - Doelman - Exponential Convergence of The Galerkin
                Approximation for the Ginzburg-Landau Equation
        |   [3] Peyret, Roger. Spectral methods for incompressible viscous
                flow. Vol. 148. New York: Springer, 2002, ch. 2.
        """
        # initialize the superclass
        super().__init__()
        # initialize the model parameters
        self.K = K
        self.N = 2 * K + 1
        self.alpha = alpha_r + 1j * alpha_i
        self.L = L
        self.beta = beta_r + 1j * beta_i
        self.gamma = gamma_r + 1j * gamma_i
        self._K_prime = 3 * K // 2
        self._N_prime = 2 * self._K_prime + 1
        k = cat((arange(0, K + 1), arange(-K, 0)))
        diagonal = self.alpha - self.beta * (2 * pi * k / self.L)**2
        self._A = Parameter(zeros((self.dims_state[-1], self.dims_state[-1])),
                            requires_grad=False)
        for i, d in enumerate(diagonal):
            self._A[i, i] = d.real
            self._A[i, i + self.N] = -d.imag
            self._A[i + self.N, i] = d.imag
            self._A[i + self.N, i + self.N] = d.real

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) +(2 * (2 * self.K + 1),)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) +(2 * (2 * self.K + 1),)``

        | **Raises**
        |   None

        | **References**
        |   [1] 1988 - Doering - Low-Dimensional Behavior In the Complex
                Ginzburg-Landau Equation
        |   [2] 1993 - Doelman - Exponential Convergence of The Galerkin
                Approximation for the Ginzburg-Landau Equation
        |   [3] Peyret, Roger. Spectral methods for incompressible viscous
                flow. Vol. 148. New York: Springer, 2002, ch. 2.
        """
        U = x[..., :self.N] + 1j * x[..., self.N:]
        u = self._dealiased_ifft(U)
        U_dot = self._dealiased_fft(self.gamma * conj(u) * u * u)
        return cat((U_dot.real, U_dot.imag), dim=-1)

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`.

        This method returns an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim
        U[[0, 1)]`. In particular, an initial condition is sampled in the
        following way: First,
        :math:`\\vec{u} \\sim U[\\mathcal{S}^{2(2 K + 1) - 1}]`. Second,
        :math:`r\\sim U[[0, 1)]`. Finally, the sample
        :math:`r\\vec{u}` is returned.


        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(2 * (2 * self.K + 1),)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the initial condition where ||x||_2 ~ U[[0, 1)]
        gaussian = randn((2 * (2 * self.K + 1),),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = rand((1,), device=next(self.parameters()).device.type)
        return radius * direction

    def state_to_phys(self, x: Tensor) -> Tensor:
        """Return the physical state given the state.

        This method returns the physical state given the state.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(2 * (2 * self.K + 1),)``

        | **Returns**
        |   ``Tensor``: the physical state with shape
                ``(...,) + (self.N,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        U = x[..., :self.N] + 1j * x[..., self.N:]
        return ifft(U, n=self.N, norm='forward')

    def phys_to_state(self, u: Tensor) -> Tensor:
        """Return the state given the physical state.

        This method returns the state given the physical state.

        | **Args**
        |   ``u`` (``Tensor``): the physical state with shape
                ``(...,) + (self.N,)``

        | **Returns**
        |   ``Tensor``: the state with shape
                ``(...,) + (2 * (2 * self.K + 1),)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        U = fft(u, n=self.N, norm='forward')
        return cat((U.real, U.imag), dim=-1)

    def _dealiased_ifft(self, U: Tensor) -> Tensor:
        shape = U.shape[:-1] + (self._N_prime - (self.K + 1) - self.K,)
        device = next(self.parameters()).device.type
        dtype = U.dtype
        U_pad = cat((U[..., :self.K + 1],
                     zeros(shape, dtype=dtype, device=device),
                     U[..., -self.K:]), dim=-1)
        return ifft(U_pad, n=self._N_prime, norm='forward')

    def _dealiased_fft(self, u: Tensor) -> Tensor:
        U_pad = fft(u, n=self._N_prime, norm='forward')
        neg_modes = U_pad[..., -self.K:]
        pos_modes = U_pad[..., :self.K + 1]
        return cat((pos_modes, neg_modes), dim=-1)
