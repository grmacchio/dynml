"""Contain all code related to the Kuramoto-Sivashinsky dynamical system.

This module contains all code related to the Kuramoto-Sivashinsky dynamical
system.
"""


# import built-in python-package code
from typing import Tuple
# import external python-package code
from torch import arange, cat, diag, pi, rand, randn, Tensor, zeros
from torch.fft import irfft, rfft
from torch.linalg import norm
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# export public code
__all__ = ['KSE']


# argument parser choice dictionaries
class KSE(SemiLinearFirstOrderSystem):
    """Represent the discretized Kuramoto-Sivashinsky dynamical system with
    periodic boundary conditions.

    This class represents the discretized Kuramoto-Sivashinsky dynamical system
    with periodic boundary conditions. The Kuramoto-Sivashinsky equation is
    a partial differential equation of the form

    .. math::
        \\begin{align*}
            \\frac{\\partial u}{\\partial t} + \\left(
            \\frac{\\partial^2}{\\partial x^2} + \\frac{\\partial^4}
            {\\partial x^4}\\right)u + \\frac{\\partial u}{\\partial x} u = 0,
        \\end{align*}

    where periodic boundary conditions are satisfied,
    :math:`x \\in [0, L]`, and :math:`t \\in \\mathbb{R}`.
    In order to obtain a first-order O.D.E. system, we orthogonally
    project this P.D.E. onto the space of Fourier modes where solutions
    :math:`u` take the form,

    .. math::
        \\begin{align*}
            u = \\sum_{k=-K}^{K} e^{\\frac{2\\pi i k}{L} \\:\\cdot_x} \\:
            U_k(\\:\\cdot_{t}).
        \\end{align*}

    By substituting this expression into the P.D.E. and projecting using the
    the :math:`1/L` forward-normalized Fourier transform
    :math:`\\mathcal{F}_{k}` on :math:`[0, L]`, we can write
    the K.S.E. as a system of complex-valued ordinary differential equations:

    .. math::
        \\begin{align*}
            \\dot{U}_{k} &= \\left(\\left(\\frac{2\\pi k}{L}\\right)^2
            - \\left(\\frac{2\\pi k}{L}\\right)^4\\right) U_{k}
            - \\mathcal{F}_k \\left(\\frac{\\partial u}{\\partial x} u\\right)
        \\end{align*}

    for :math:`k \\in [-K : K]`. The nonlinear term is
    approximated using the :math:`1/(2K + 1)` forward-normalized discrete
    Fourier transform :math:`\\mathcal{D}_k` on :math:`[-K : K]` and applying
    the nonlinearity to the solution's values at the collocation points:

    .. math::
        \\begin{align*}
            \\mathcal{F}_{k}\\left(\\partial u / \\partial x \\: u\\right)
            \\approx
            \\mathcal{D}_{k}
            \\left[
            \\left(
            \\mathcal{D}_{n}^{-1}
            \\left[
            \\left(
            \\frac{2\\pi i m}{L} U_m(\\:\\cdot_{t})
            \\right)_{m\\:}
            \\right]
            \\cdot
            \\mathcal{D}_{n}^{-1}
            \\left[\\left(
            U_{m}(\\:\\cdot_{t}) \\right)_{m\\:}
            \\right]
            \\right)_{n\\:}\\right],
        \\end{align*}

    where :math:`m \\in [-K : K]` is the frequency index and
    :math:`n \\in [0 : 2K]` is the collocation point index [1]. When using the
    discrete fourier transform we account for aliasing by
    using the :math:`3/2` rule [1]. Finally, we represent this system as
    smaller system of ordinary differential equations by removing any
    extraneous states in :math:`(U_{k}(\\:\\cdot_{t}))_{k\\in [-K, K]}` using
    the Hermitian symmetry condition of real-valued signals. The final system
    takes the form,

    .. math::
        \\begin{align*}
            \\dot{\\vec{x}} = A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    where

    .. math::
        \\begin{align*}
            \\vec{x} &= (\\text{Re} \\: U_{0}, \\text{Re} \\: U_1, \\cdots,
            \\text{Re} \\: U_K, \\text{Im} \\: U_1, \\cdots, \\text{Im} \\
            U_{K}).
        \\end{align*}

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): ``R`` for real numbers
    |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions
    |   ``K`` (``int``): the number of Fourier modes :math:`K` with default
            value ``32``
    |   ``L`` (``float``): the length of the domain :math:`L` with a default
            value ``11.0``
    |   ``N`` (``int``): the number of collocation points :math:`2K + 1`
    |   ``A`` (``Tensor``): the matrix :math:`A`

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``nonlinear()``: return :math:`F(\\vec{x})`
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``state_to_phys()``: return the physical state given the state
    |   ``phys_to_state()``: return the state given the physical state
    |   ``gen_ic()``: Return an I.C. where
        :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`
    |   ``rhs()``: return :math:`f(\\vec{x})`

    | **References**
    |   [1] Peyret, Roger. Spectral methods for incompressible viscous flow.
            Vol. 148. New York: Springer, 2002, ch. 2.
    """

    @property
    def field(self) -> str:
        return 'R'

    @property
    def dims_state(self) -> Tuple[int, ...]:
        return (2 * self.K + 1,)

    def __init__(self, K: int = 32, L: float = 11.0):
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``K`` (``int``): the number of Fourier modes :math:`K` with a
                default value of ``32``
        |   ``L`` (``float``): the length of the domain :math:`L` with a
                default value of ``11.0``

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__()
        # initialize the model parameters
        self.K = K
        self.L = L
        self.N = 2 * K + 1
        self._K_prime = 3 * self.K // 2
        self._N_prime = 2 * self._K_prime + 1
        k = cat((arange(0, self.K + 1), arange(1, self.K + 1)))
        ksq = (2 * pi / L) ** 2 * k**2
        self._A = Parameter(diag(ksq * (1 - ksq)), requires_grad=False)
        self._freq_deriv = Parameter((2j * pi / L) * k, requires_grad=False)

    @property
    def A(self) -> Tensor:
        return self._A

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral methods for incompressible viscous
                flow. Vol. 148. New York: Springer, 2002, ch. 2.
        """
        # represent the state in frequency space
        U = x[..., :self.K + 1]
        U[1:] = U[1:] + 1j * x[..., self.K + 1:]
        u = self._dealiased_irfft(U)
        u_x = self._dealiased_irfft(U * self._freq_deriv)
        U_dot = -1 * self._dealiased_rfft(u * u_x)
        return cat((U_dot.real, U_dot.imag[1:]), dim=-1)

    def state_to_phys(self, x: Tensor) -> Tensor:
        """Return the physical state given the state.

        This method returns the physical state given the state.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the physical state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        U = x[..., :self.K + 1]
        U[1:] = U[1:] + 1j * x[..., self.K + 1:]
        return irfft(U, n=self.N, norm='forward')

    def phys_to_state(self, u: Tensor) -> Tensor:
        """Return the state given the physical state.

        This method returns the state given the physical state.

        | **Args**
        |   ``u`` (``Tensor``): the physical state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        U = rfft(u, n=self.N, norm='forward')
        return cat((U.real, U.imag[1:]), dim=-1)

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`.

        This method returns an I.C. where
        :math:`\\|\\vec{x\\|_2 \\sim U[[0, 1)]`. In particular,
        an initial condition is sampled in the following way: First,
        :math:`\\vec{u} \\sim U[\\mathcal{S}^{2 K}]`. Second,
        :math:`r\\sim U[[0, 1)]`. Finally, the sample
        :math:`\\vec{x} = ru` is returned.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        gaussian = randn((2 * self.K + 1,),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = rand((1,), device=next(self.parameters()).device.type)
        return radius * direction

    def _dealiased_irfft(self, U: Tensor) -> Tensor:
        U_pad = cat((U[..., :self.K + 1],
                     zeros(U.shape[:-1] + (self._K_prime - self.K,),
                           dtype=U.dtype,
                           device=next(self.parameters()).device.type)),
                    dim=-1)
        return irfft(U_pad, n=self._N_prime, norm='forward')

    def _dealiased_rfft(self, u: Tensor) -> Tensor:
        return rfft(u, n=self._N_prime, norm='forward')[..., :self.K + 1]
