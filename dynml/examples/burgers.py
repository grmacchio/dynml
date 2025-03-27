"""Contain all code related to the Burgers' dynamical system.

This module contains all code related to the Burgers' dynamical system.
"""


# import built-in python-package code
from math import pi as math_pi
# import external python-package code
from torch import arange, diag, pi, rand, randn, Tensor, zeros
from torch.fft import irfft, rfft
from torch.linalg import norm
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# export public code
__all__ = ['Burgers']


# define the dynamical system
class Burgers(SemiLinearFirstOrderSystem):
    """Represent the discretized Burgers' dynamical system with periodic
    boundary conditions.

    This class represents the discretized Burgers' dynamical system with
    periodic boundary conditions on :math:`[0, L]`. Burgers' equation
    is written as

    .. math::
        \\frac{\\partial u}{\\partial t} =
        \\nu \\frac{\\partial^2 u}{\\partial x^2}
        - u \\frac{\\partial u}{\\partial x},

    where :math:`u(t, x) \\in \\mathbb{R}` is the velocity field, :math:`t \\in
    \\mathbb{R}` is the time, :math:`x \\in [0, L]` is the spatial
    coordinate, and :math:`\\nu > 0` is the viscosity. In order to obtain a
    first-order O.D.E. system, we orthogonally project this P.D.E. onto the
    space of complex Fourier modes where solutions :math:`u` take the form,

    .. math::
        \\begin{align*}
            u = \\sum_{k=-K}^{K} e^{\\frac{2\\pi i k}{L} \\:\\cdot_x} \\:
            U_k(\\:\\cdot_{t}).
        \\end{align*}

    By substituting this expression into the P.D.E. and projecting using the
    the :math:`1/L` forward-normalized Fourier transform
    :math:`\\mathcal{F}_{k}` on :math:`[0, L]`, we can write
    the Burgers' equation as a system of complex-valued ordinary differential
    equations:

    .. math::
        \\begin{align*}
            \\dot{U}_{k} &= -\\nu \\left(\\frac{2 \\pi k}{L}\\right)^2 U_{k}
            - \\mathcal{F}_k \\left(u \\frac{\\partial u}{\\partial x}\\right).
        \\end{align*}

    for :math:`k \\in [-K : K]`. The nonlinear term is approximated using the
    :math:`1 / (2K + 1)` forward-normalized discrete Fourier transform
    :math:`\\mathcal{D}_k` on :math:`[-K : K]` and applying the nonlinearity
    to the solution's values at the collocation points:

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
            \\frac{2\\pi i m}{L}
            U_m
            \\right)_{m\\:}
            \\right]
            \\cdot
            \\mathcal{D}_{n}^{-1}
            \\left[
            \\left(
            U_{m}
            \\right)_{m\\:}
            \\right]
            \\right)_{n\\:}\\right],
        \\end{align*}

    where :math:`m \\in [-K : K]` is the frequency index and
    :math:`n \\in [0 : 2K]` is the collocation point index [1]. When using the
    discrete fourier transform we account for aliasing by
    using the :math:`3/2` rule [1]. Finally, we represent this system as
    smaller system of ordinary differential equations by removing any
    extraneous states in :math:`(U_{k}(\\:\\cdot_{t}))_{k\\in [-K, K]}` using
    the Hermitian symmetry condition of real-valued signals. The left over
    complex-valued states are :math:`\\vec{x} = (U_{k}(\\:\\cdot_{t})
    )_{k \\in [0 : K]}`. The final system takes the
    form,

    .. math::
        \\begin{align*}
            \\dot{\\vec{x}} = A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    One could reduce the memory requirement by noting :math:`U_{0}` is real and
    removing the one redundant state, resulting in
    :math:`\\vec{x}_{\\mathbb{R}}` real-valued state. This, was not implemented
    as the memory savings are minimal and reshaping into real tensors would
    require more overhead than its worth considering states in the frequency
    domain, or physical domain, are readily used in computation.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``num_states`` (``int``): the number of states
    |   ``A`` (``Tensor``): the matrix :math:`A`
    |   ``K`` (``int``): the number of Fourier modes :math:`K` with a default
            value of ``512``
    |   ``L`` (``float``): the length of the domain :math:`L` with a default
            value of ``2.0 * pi``
    |   ``nu`` (``float``): the viscosity :math:`\\nu` with a default value
            of ``0.01``
    |   ``N`` (``int``): the number of collocation points :math:`2K + 1`

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``nonlinear()``: return :math:`F(\\vec{x})`
    |   ``state_to_phys()``: return the physical state given the state
    |   ``phys_to_state()``: return the state given the physical state
    |   ``gen_ic()``: Return an I.C. where
        :math:`\\|\\vec{x}_{\\mathbb{R}}\\|_2 \\sim U[[0, 1)]`
    |   ``rhs()``: return :math:`f(\\vec{x})`

    | **References**
    |   [1] Peyret, Roger. Spectral methods for incompressible viscous flow.
            Vol. 148. New York: Springer, 2002, ch. 2.
    """

    @property
    def num_states(self):
        return self.K + 1

    @property
    def A(self) -> Tensor:
        return self._A

    def __init__(self, K: int = 512, L: float = 2.0 * math_pi,
                 nu: float = 0.01) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``K`` (``int``): the number of Fourier modes :math:`K` with a
                default value of ``512``
        |   ``L`` (``float``): the length of the domain :math:`L`
                with a default value of ``2.0 * pi``
        |   ``nu`` (``float``): the viscosity :math:`\\nu` with a default
                value of ``0.01``

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
        self.nu = nu
        self.N = 2 * K + 1
        self._K_prime = 3 * self.K // 2
        self._N_prime = 2 * self._K_prime + 1
        k = arange(self.K + 1)
        self._A = Parameter(diag(self.nu * -1 * (2 * pi / L) ** 2 * k**2),
                            requires_grad=False)
        self._freq_deriv = Parameter((2j * pi / L) * k, requires_grad=False)

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``\\vec{x}`` (``Tensor``): the state with shape
                ``(...,) + (self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) + (self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral methods for incompressible viscous
                flow. Vol. 148. New York: Springer, 2002, ch. 2.
        """
        # approximate the nonlinear term
        u = self._dealiased_irfft(x)
        u_x = self._dealiased_irfft(x * self._freq_deriv)
        return -1 * self._dealiased_rfft(u * u_x)

    def state_to_phys(self, x: Tensor) -> Tensor:
        """Return the physical state given the state.

        This method returns the physical state given the state.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape ``(...,) + (self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the physical state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return irfft(x, n=self.N, norm='forward')

    def phys_to_state(self, u: Tensor) -> Tensor:
        """Return the state given the physical state.

        This method returns the state given the physical state.

        | **Args**
        |   ``u`` (``Tensor``): the physical state with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the state with shape
                ``(...,) + (self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return rfft(u, n=self.N, norm='forward')

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}_{\\mathbb{R}}\\|_2
        \\sim U[[0, 1)]`.

        This method returns an I.C. where
        :math:`\\|\\vec{x}_{\\mathbb{R}}\\|_2 \\sim U[[0, 1)]`. In particular,
        an initial condition is sampled in the following way: First,
        :math:`\\vec{u} \\sim U[\\mathcal{S}^{2n-2}]`. Second,
        :math:`r\\sim U[[0, 1)]`. Finally, the sample
        :math:`\\vec{x}_{\\mathbb{R}} = ru` is returned shaped into the state
        :math:`\\vec{x}`.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(...,) + (self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        gaussian = randn((2 * self.K + 1,),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = rand((1,), device=next(self.parameters()).device.type)
        point = radius * direction
        real = zeros((self.K + 1,), device=next(self.parameters()).device.type)
        comp = zeros((self.K + 1,), device=next(self.parameters()).device.type)
        real = point[:self.K + 1]
        comp[1:] = point[self.K + 1:]
        return real + 1j * comp

    def _dealiased_irfft(self, x: Tensor) -> Tensor:
        x_pad = zeros(x.shape[:-1] + (self._K_prime + 1,),
                      dtype=x.dtype,
                      device=next(self.parameters()).device.type).squeeze(0)
        x_pad[..., :self.K + 1] = x[..., :self.K + 1]
        return irfft(x_pad, n=self._N_prime, norm='forward')

    def _dealiased_rfft(self, u: Tensor) -> Tensor:
        return rfft(u, n=self._N_prime, norm='forward')[..., :self.K + 1]
