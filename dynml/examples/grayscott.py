"""Contain all code related to the Gray-Scott system.

This module contains all code related to the Gray-Scott system.
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
__all__ = ['GrayScott']


# define the dynamical system
class GrayScott(SemiLinearFirstOrderSystem):
    """Represent the discretized Gray-Scott dynamical system with periodic
    boundary conditions.

    This class represents the discretized Gray-Scott dynamical system with
    periodic boundary conditions. This system is a canonical example of a
    reaction-diffusion process that describes a cubic autocatalytic mechanism
    in an open reactor, which is a reactor whose chemicals are continuously
    supplied, removed, and allowed to diffuse while reacting [1, 2]. In
    particular, this system corresponds to the following set of reactions:

    .. math::
        \\begin{align*}
            A + 2B &\\rightarrow 3B
            \\\\
            B &\\rightarrow C,
        \\end{align*}

    where :math:`A` is called the fed reactant, :math:`B` is called the
    autocatalyst, and :math:`C` is called the inert product. As a dimensionless
    partial differential equation, the system takes the form

    .. math::
        \\begin{align*}
            \\frac{\\partial a}{\\partial t}
            &=
            \\alpha \\frac{\\partial^2 a}{\\partial x^2}
            + f \\left(1 - a \\right)
            - a b^2
            \\\\
            \\frac{\\partial b}{\\partial t}
            &=
            \\beta \\frac{\\partial^2 b}{\\partial x^2}
            - f b - k b
            + a b^2,
        \\end{align*}

    where :math:`a(t, x), b(t, x), \\alpha, \\beta, f, k \\in [0, \\infty)`,
    :math:`t \\in \\mathbb{R}`, and :math:`x \\in \\left[0, L\\right]`. One
    could define :math:`x` to be higher-dimensional; however, here, we focus on
    the one-dimensional case. Note, there exists a fixed point at
    :math:`(1, 0)`; consequently, we transform the system using
    :math:`u = a - 1` and :math:`v = b`:

    .. math::
        \\begin{align*}
            \\frac{\\partial u}{\\partial t}
            &=
            \\alpha \\frac{\\partial^2 u}{\\partial x^2}
            - f u
            - v^2
            - u v^2
            \\\\
            \\frac{\\partial v}{\\partial t}
            &=
            \\beta \\frac{\\partial^2 v}{\\partial x^2}
            - f v - k v
            + v^2
            + u v^2.
        \\end{align*}

    In order to obtain a first-order O.D.E. system, we orthogonally project
    this P.D.E. onto the space of complex Fourier modes where solutions
    :math:`(u, v)` take the form

    .. math::
        \\begin{align*}
            u = \\sum_{k=-K}^{K} e^{\\frac{2\\pi i k}{L} \\:\\cdot_x} \\:
            U_k(\\:\\cdot_{t})
            \\quad \\text{ and } \\quad
            v = \\sum_{k=-K}^{K} e^{\\frac{2\\pi i k}{L} \\:\\cdot_x} \\:
            V_k(\\:\\cdot_{t}).
        \\end{align*}

    By substituting this expression into the P.D.E. and projected using the
    :math:`1 / L` forward-normalized Fourier transform :math:`\\mathcal{F}_k`
    on :math:`[0, L]`, we can write the discretized Gray-Scott equations as a
    system of complex-valued ordinary differential equations:

    .. math::
        \\begin{align*}
            \\dot{U}_{k}
            &=
            \\left(
            - \\alpha \\left(\\frac{2\\pi k}{L} \\right)^2
            - f
            \\right) U_k
            - \\mathcal{F}_{k} \\left[v^2 + u v^2 \\right]
            \\\\
            &=
            \\left(
            - \\alpha \\left(\\frac{2\\pi k}{L} \\right)^2
            - f
            \\right) U_k
            - \\sum_{\\underset{k_1 + k_2 = k}
            {k_1, k_2 \\in \\mathbb{Z}}} V_{k_1} V_{k_2}
            - \\sum_{\\underset{k_1 + k_2 + k_3 = k}
            {k_1, k_2, k_3 \\in \\mathbb{Z}}} U_{k_1} V_{k_2} V_{k_3}
            \\\\
            \\dot{V}_{k}
            &=
            \\left(
            - \\beta \\left(\\frac{2\\pi k}{L} \\right)^2
            - f
            - k
            \\right) V_k
            + \\mathcal{F}_{k} \\left[v^2 + u v^2 \\right]
            \\\\
            &=
            \\left(
            - \\beta \\left(\\frac{2\\pi k}{L} \\right)^2
            - f
            - k
            \\right) V_k
            + \\sum_{\\underset{k_1 + k_2 = k}
            {k_1, k_2 \\in \\mathbb{Z}}} V_{k_1} V_{k_2}
            + \\sum_{\\underset{k_1 + k_2 + k_3 = k}
            {k_1, k_2, k_3 \\in \\mathbb{Z}}} U_{k_1} V_{k_2} V_{k_3},
        \\end{align*}

    In ``nonlinear()``, we approximate the nonlinear term using the
    :math:`1 / (2K + 1)` forward-normalized discrete Fourier transform
    :math:`\\mathcal{D}_k` on :math:`[-K : K]` and applying the nonlinearity to
    the solution's values at the collocation points:

    .. math::
        \\begin{align*}
            \\mathcal{F}_{k}
            \\left[
            v^2
            + u v^2
            \\right]
            &\\approx
            \\mathcal{D}_{k}\\left[
            \\left(
                \\mathcal{D}_{n}^{-1} \\left[(V_{m})_{m\\:}\\right]
                \\cdot
                \\mathcal{D}_{n}^{-1} \\left[(V_{m})_{m\\:}\\right]
            \\right)_{n\\:}
            \\right]
            \\\\
            &\\quad\\:\\:+
            \\mathcal{D}_{k}\\left[
            \\left(
                \\mathcal{D}_{n}^{-1} \\left[(U_{m})_{m\\:}\\right]
                \\cdot
                \\mathcal{D}_{n}^{-1} \\left[(V_{m})_{m\\:}\\right]
                \\cdot
                \\mathcal{D}_{n}^{-1} \\left[(V_{m})_{m\\:}\\right]
            \\right)_{n\\:}
            \\right].
        \\end{align*}

    where :math:`m \\in [-K : K]` is the frequency index and
    :math:`n \\in [0 : 2K]` is the collocation point index [3]. When using the
    discrete fourier transform we account for aliasing by using the
    :math:`2 \\times` rule [3]. Finally, we represent this system as a smaller
    system of ordinary differential equations by removing any extraneous states
    in :math:`(U_{k}, V_{k})_{k \\in [-K, K]}` using the Hermitian symmetry
    condition of real-valued signals. The final system takes the form,

    .. math::
        \\begin{align*}
            \\dot{\\vec{x}} = A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    where

    .. math::
        \\begin{align*}
            \\vec{x}
            &=
            (\\text{Re} \\: U_0, \\ldots, \\text{Re} \\: U_K,
            \\text{Im} \\: U_1, \\ldots, \\text{Im} \\: U_{K},
            \\text{Re} \\: V_0, \\ldots, \\text{Re} \\: V_K,
            \\text{Im} \\: V_1, \\ldots, \\text{Im} \\: V_{K}).
        \\end{align*}

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``K`` (``int``): the number of Fourier modes :math:`K` with default
            value ``32``
    |   ``L`` (``float``): the value of the parameter :math:`L` with a default
            value ``20.0``
    |   ``alpha`` (``float``): the value of the parameter :math:`\\alpha` with
            a default value ``1e-2``
    |   ``beta`` (``float``): the value of the parameter :math:`\\beta` with a
            default value ``1e-2``
    |   ``f`` (``float``): the value of the parameter :math:`f` with a default
            value ``0.04``
    |   ``k`` (``float``): the value of the parameter :math:`k` with a default
            value ``0.24``
    |   ``N`` (``int``): the number of collocation points :math:`2K + 1`
    |   ``dims_state`` (``Tuple[int, ...]``): the state dimensions
    |   ``A`` (``Tensor``): the matrix :math:`A`

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``nonlinear()``: return :math:`F(\\vec{x})`
    |   ``rhs()``: return :math:`f(\\vec{x})`
    |   ``gen_ic()``: generate initial conditions where
            :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`
    |   ``state_to_phys()``: return the physical state given the state
    |   ``phys_to_state()``: return the state given the physical state

    | **References**
    |   [1] Gray, Peter, and Stephen K. Scott. "Autocatalytic reactions in the
            isothermal, continuous stirred tank reactor: isolas and other forms
            of multistability." Chemical Engineering Science 38, no. 1 (1983):
            29-43.
    |   [2] Pearson, John E. "Complex patterns in a simple system." Science
            261, no. 5118 (1993): 189-192.
    |   [3] Peyret, Roger. Spectral methods for incompressible viscous flow.
            Vol. 148. New York: Springer, 2002, ch. 2.
    """

    @property
    def dims_state(self):
        return (2 * (2 * self.K + 1),)

    @property
    def A(self):
        return self._A

    def __init__(self, K: int = 32, L: float = 20, alpha: float = 1e-2,
                 beta: float = 1e-2, f: float = 0.04, k: float = 0.24) -> None:
        """Initialize the superclass and model parameters.

        This method initialize the superclass and model parameters.

        | **Args**
        |   ``K`` (``int``): the number of Fourier modes :math:`K` with default
                value ``32``
        |   ``L`` (``float``): the value of the parameter :math:`L` with a
                default value ``20.0``
        |   ``alpha`` (``float``): the value of the parameter :math:`\\alpha`
                with a default value ``1e-2``
        |   ``beta`` (``float``): the value of the parameter :math:`\\beta`
                with a default value ``1e-2``
        |   ``f`` (``float``): the value of the parameter :math:`f` with a
                default value ``0.04``
        |   ``k`` (``float``): the value of the parameter :math:`k` with a
                default value ``0.24``

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
        self.alpha = alpha
        self.beta = beta
        self.f = f
        self.k = k
        self.K = K
        self.L = L
        self.N = 2 * K + 1
        self._K_prime = 2 * K
        self._N_prime = 2 * self._K_prime + 1
        ks = cat((arange(0, K + 1), arange(1, K + 1)))
        ksq = (2 * pi / self.L)**2 * ks**2
        diag_A_u = -self.alpha * ksq - self.f
        diag_A_v = -self.beta * ksq - self.f - self.k
        diag_A = cat((diag_A_u, diag_A_v))
        self._A = Parameter(diag(diag_A), requires_grad=False)

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) + (2 * (2 * self.K + 1),)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) + (2 * (2 * self.K + 1),)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral methods for incompressible viscous
                flow. Vol. 148. New York: Springer, 2002, ch. 2.
        """
        # convert to collocation space
        U = zeros(x.shape[:-1] + (1 + self.K,),
                  device=next(self.parameters()).device.type) + 1j * 0.0
        U[..., 0] = x[..., 0] + 1j * 0.0
        U[..., 1:] = (x[..., 1:self.K + 1]
                      + 1j * x[..., self.K + 1:2 * self.K + 1])
        V = zeros(x.shape[:-1] + (1 + self.K,),
                  device=next(self.parameters()).device.type) + 1j * 0.0
        V[..., 0] = x[..., 2 * self.K + 1] + 1j * 0.0
        V[..., 1:] = (x[..., 2 * self.K + 2:3 * self.K + 2]
                      + 1j * x[..., 3 * self.K + 2:])
        u = self._dealiased_irfft(U)
        v = self._dealiased_irfft(V)
        # compute in collocation space
        output = self._dealiased_rfft(v * v + u * v * v)
        # convert to frequency space
        return cat((-1 * output.real, -1 * output[..., 1:].imag,
                    output.real, output[..., 1:].imag), dim=-1)

    def state_to_phys(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return the physical state given the state.

        This method returns the physical state given the state.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) + (2 * (2 * self.K + 1),)``

        | **Returns**
        |   ``Tuple[Tensor, Tensor]``: the physical state :math:`(u, v)` each
                with shape ``(...,) + (2 * self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # convert to collocation space
        U = zeros(x.shape[:-1] + (1 + self.K,),
                  device=next(self.parameters()).device.type) + 1j * 0.0
        U[..., 0] = x[..., 0] + 1j * 0.0
        U[..., 1:] = (x[..., 1:self.K + 1]
                      + 1j * x[..., self.K + 1:2 * self.K + 1])
        V = zeros(x.shape[:-1] + (1 + self.K,),
                  device=next(self.parameters()).device.type) + 1j * 0.0
        V[..., 0] = x[..., 2 * self.K + 1] + 1j * 0.0
        V[..., 1:] = (x[..., 2 * self.K + 2:3 * self.K + 2]
                      + 1j * x[..., 3 * self.K + 2:])
        return (irfft(U, n=self.N, norm='forward'),
                irfft(V, n=self.N, norm='forward'))

    def phys_to_state(self, u: Tensor, v: Tensor) -> Tensor:
        """Return the state given the physical state.

        This method returns the state given the physical state.

        | **Args**
        |   ``u`` (``Tensor``): the physical state :math:`u` with shape
                ``(...,) + (2 * self.K + 1,)``
        |   ``v`` (``Tensor``): the physical state :math:`v` with shape
                ``(...,) + (2 * self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the state with shape
                ``(...,) + (2 * (2 * self.K + 1),)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # convert to state space
        U = rfft(u, n=self.N, norm='forward')
        V = rfft(v, n=self.N, norm='forward')
        return cat((U.real, U[..., 1:].imag, V.real, V[..., 1:].imag), dim=-1)

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`.

        This method returns an I.C. where
        :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`. In particular,
        an initial condition is sampled in the following way: First,
        :math:`\\vec{u} \\sim U[\\mathcal{S}^{2 K}]`. Second,
        :math:`r\\sim U[[0, 1)]`. Finally, the sample
        :math:`\\vec{x} = ru` is returned. Note, the returned :math:`\\vec{x}`
        is not physical, unless, one guarantees that the corresponding
        functional initial condition :math:`(u_0, v_0)` satisfies
        :math:`u_0(x) \\geq -1` and :math:`v_0(x) \\geq 0`.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(...,) + (2 * (2 * self.K + 1),)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        gaussian = randn((2 * (2 * self.K + 1),),
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
