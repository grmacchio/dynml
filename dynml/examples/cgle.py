"""Contain all code related to the complex Ginzburg-Landau dynamical system.

This module contains all code related to the complex Ginzburg-Landau dynamical
system.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import arange, cat, conj, diag, pi, rand, randn
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
        = \\left(R + (1 + i \\nu) \\frac{\\partial^2}{\\partial x^2}\\right)u
        - (1 + i \\mu) \\left|u\\right|^2 u

    where :math:`u(t, x) \\in \\mathbb{C}`,
    :math:`t, \\nu, \\mu \\in\\mathbb{R}`, and :math:`x \\in [0, L]`. In order
    to obtain a first-order O.D.E. system, we orthogonally project this P.D.E.
    onto the space of complex Fourier modes where solutions :math:`u` take the
    form,

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
            \\dot{U}_{k} &= \\left(R - (1 + i \\nu) \\left(\\frac{2\\pi k}{L}
            \\right)^2\\right) U_k
            + \\mathcal{F}_{k} \\left( - (1 + i \\mu) |u|^2 u \\right)
        \\end{align*}

    for :math:`k \\in [-K : K]`. The nonlinear term is approximated using
    the :math:`1 / (2K + 1)` forward-normalized discrete Fourier transform
    :math:`\\mathcal{D}_k` on :math:`[-K : K]` and applying the nonlinearity to
    the solution's values at the collocation points:

    .. math::
        \\begin{align*}
            \\mathcal{F}_{k}\\left(- (1 + i \\mu) |u|^2 u\\right)
            \\approx
            \\mathcal{D}_{k}\\left[
                \\left(
                -(1 + i \\mu) \\cdot
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
    :math:`3/2` rule [3]. Letting :math:`\\vec{x} = (U_{k})_{k\\in [-K : K]}`,
    the final system takes the form,

    .. math::
        \\begin{align*}
            \\dot{\\vec{x}} = A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): ``C`` for complex numbers
    |   ``num_states`` (``int``): the number of states
    |   ``A`` (``Tensor``): the matrix :math:`A`
    |   ``K`` (``int``): the number of Fourier modes :math:`K` with default
            value of ``256``
    |   ``N`` (``int``): the number of collocation points :math:`N = 2K + 1`
    |   ``L`` (``float``): the value of the parameter :math:`L` with default
            value of ``100.0``
    |   ``R`` (``float``): the value of the parameter :math:`R` with default
            value of ``1.0``
    |   ``nu`` (``float``): the value of the parameter :math:`\\nu` with a
            default value of ``1.0``
    |   ``mu`` (``float``): the value of the parameter :math:`\\mu` with a
            default value of ``2.0``

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
        return 'C'

    @property
    def A(self) -> Tensor:
        return self._A

    @property
    def num_states(self) -> int:
        return 2 * self.K + 1

    def __init__(self, K: int = 256, L: float = 100.0,
                 R: float = 1.0, nu: float = 1.0, mu: float = 2.0) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``K`` (``int``): the number of Fourier modes :math:`K` with a
                default value of ``256``
        |   ``L`` (``float``): the length of the domain :math:`L` with a
                default value of ``100.0``
        |   ``R`` (``float``): the value of the parameter :math:`R` with a
                default value of ``1.0``
        |   ``nu`` (``float``): the value of the parameter :math:`\\nu` with a
                default value of ``1.0``
        |   ``mu`` (``float``): the value of the parameter :math:`\\mu` with a
                default value of ``2.0``

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
        self.L = L
        self.nu = nu
        self.mu = mu
        self._K_prime = 3 * K // 2
        self._N_prime = 2 * self._K_prime + 1
        k = cat((arange(0, K + 1), arange(-K, 0)))
        self._A = Parameter(diag(R - (1 + 1j * self.nu)
                                 * (2 * pi * k / self.L)**2),
                            requires_grad=False)

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) +(self.N,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) +(self.N,)``

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
        u = self._dealiased_ifft(x)
        return self._dealiased_fft(- (1 + 1j * self.mu) * conj(u) * u * u)

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 1)]`.

        This method returns an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim
        U[[0, 1)]`. In particular, an initial condition is sampled in the
        following way: First, :math:`\\vec{u} \\sim U[\\mathcal{S}^{2n-1}]`.
        Second, :math:`r\\sim U[[0, 1)]`. Finally, the sample
        :math:`r\\vec{u}` is returned.


        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape ``(self.N,)``

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
        sample = radius * direction
        real = sample[:self.N]
        comp = sample[self.N:]
        return real + 1j * comp

    def state_to_phys(self, x: Tensor) -> Tensor:
        """Return the physical state given the state.

        This method returns the physical state given the state.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape ``(...,) + (self.N,)``

        | **Returns**
        |   ``Tensor``: the physical state with shape
                ``(...,) + (self.N,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return ifft(x, n=self.N, norm='forward')

    def phys_to_state(self, u: Tensor) -> Tensor:
        """Return the state given the physical state.

        This method returns the state given the physical state.

        | **Args**
        |   ``u`` (``Tensor``): the physical state with shape
                ``(...,) + (self.N,)``

        | **Returns**
        |   ``Tensor``: the state with shape
                ``(...,) + (self.N,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return fft(u, n=self.N, norm='forward')

    def _dealiased_ifft(self, x: Tensor) -> Tensor:
        shape = x.shape[:-1] + (self._K_prime - self.K,)
        device = next(self.parameters()).device.type
        dtype = x.dtype
        x_pad = cat((zeros(shape, dtype=dtype, device=device),
                     x[..., -self.K:self.K + 1]),
                     zeros(shape, dtype=dtype, device=device),
                     dim=-1)
        return ifft(x_pad, n=self._N_prime, norm='forward')

    def _dealiased_fft(self, u: Tensor) -> Tensor:
        U_pad = fft(u, n=self._N_prime, norm='forward')
        neg_modes = U_pad[..., -self.K:]
        pos_modes = U_pad[..., :self.K + 1]
        return cat((pos_modes, neg_modes), dim=-1)
