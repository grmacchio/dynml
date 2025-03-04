"""Contain all code related to the Kuramoto-Sivashinsky equation.

This module contains all code related to the Kuramoto-Sivashinsky equation.
"""


# import built-in python-package code
# None
# import external python-package code
from torch import arange, cat, diag, pi, rand, randn, Tensor, zeros, zeros_like
from torch.fft import irfft, rfft
from torch.linalg import norm
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# export public code
__all__ = ['KSE']


# argument parser choice dictionaries
class KSE(SemiLinearFirstOrderSystem):
    """Represent the discretized Kuramoto-Sivashinsky equation.

    This class represents the discretized Kuramoto-Sivashinsky equation
    (K.S.E.). The K.S.E. is a real-valued partial differential equation
    (P.D.E.) of the form

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
            \\mathcal{D}_{k}\\left(
            \\left\\{\\mathcal{D}_{n}^{-1}\\left(\\left\\{
            \\frac{2\\pi i m}{L} U_m(\\:\\cdot_{t})
            \\right\\}_{m}\\right)
            \\cdot \\mathcal{D}_{n}^{-1}\\left(\\left\\{
            U_{m}(\\:\\cdot_{t}) \\right\\}_{m}
            \\right)
            \\right\\}_{n}\\right),
        \\end{align*}

    where :math:`m \\in [-K : K]` is the frequency index and
    :math:`n \\in [0 : 2K]` is the collocation point index [1]. When using the
    discrete fourier transform we account for aliasing by
    using the :math:`3/2` rule [1]. Finally, we represent this system as a
    real-valued system of ordinary differential equations by removing any
    extraneous states in :math:`\\{U_{k}(\\:\\cdot_{t})\\}_{k}` using the
    Hermitian symmetry condition of real-valued signals and by reshaping the
    left over complex-valued states into a real-valued vector :math:`\\vec{x}`.
    The final system takes the form,

    .. math::
        \\begin{align*}
            \\dot{\\vec{x}} &= A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``K`` (``int``): the number of Fourier modes :math:`K` with default
            value ``32``
    |   ``L`` (``float``): the length of the domain :math:`L` with a default
            value ``22.0``
    |   ``N`` (``int``): the number of collocation points :math:`2K + 1`
    |   ``num_states`` (``int``): the number of states
    |   ``A`` (``Tensor``): the matrix :math:`A`

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``nonlinear()``: return :math:`F(\\vec{x})`
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``rfreq_to_state()``: return the state given the ``rfft``-formatted
            complex state.
    |   ``state_to_rfreq()``: return the ``rfft``-formatted complex state
            given the state.
    |   ``gen_ic()``: Return an I.C. where :math:`\\|\\vec{x}\\|_2
            \\sim U[[0, 2)]`
    |   ``rhs()``: return :math:`f(\\vec{x})`

    | **References**
    |   [1] Peyret, Roger. Spectral methods for incompressible viscous flow.
            Vol. 148. New York: Springer, 2002, ch. 2.
    """

    @property
    def num_states(self):
        return 2 * (self.K + 1) - 1

    def __init__(self, K: int = 32, L: float = 22.0):
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``K`` (``int``): the number of Fourier modes :math:`K` with a
                default value of ``32``
        |   ``L`` (``float``): the length of the domain :math:`L` with a
                default value of ``22.0``

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
        k = cat((arange(self.K + 1), arange(1, self.K + 1)))
        ksq = (2 * pi / L) ** 2 * k**2
        self._A = Parameter(diag(ksq * (1 - ksq)), requires_grad=False)
        self._freq_deriv = Parameter((2j * pi / L) * arange(self.K + 1),
                                     requires_grad=False)

    @property
    def A(self) -> Tensor:
        return self._A

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) + (2 * (self.K + 1) - 1,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) + (2 * (self.K + 1) - 1,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral methods for incompressible viscous
                flow. Vol. 148. New York: Springer, 2002, ch. 2.
        """
        # represent the state in frequency space
        U = self.state_to_rfreq(x)
        # approximate the nonlinear term
        u = self._dealiased_irfft(U)
        u_x = self._dealiased_irfft(U * self._freq_deriv)
        return -1 * self.rfreq_to_state(self._dealiased_rfft(u * u_x))

    def rfreq_to_state(self, U: Tensor) -> Tensor:
        """Return the state given the ``rfft``-formatted complex state.

        This method returns the state given the ``rfft``-formatted complex
        state.

        | **Args**
        |   ``U`` (``Tensor``): the complex state with shape
                ``(...,) + (self.K + 1,)``

        | **Returns**
        |   ``Tensor``: the real state with shape
                ``(...,) + (2 * (self.K + 1) - 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the state
        U_real = U.real
        U_imag = U.imag[..., 1:]
        return cat((U_real, U_imag), dim=-1)

    def state_to_rfreq(self, u: Tensor) -> Tensor:
        """Return the ``rfft``-formatted complex state given the state.

        This method returns the ``rfft``-formatted complex state given the
        state.

        | **Args**
        |   ``u`` (``Tensor``): the real state with shape
                ``(...,) + (2 * (self.K + 1) - 1,)``

        | **Returns**
        |   ``Tensor``: the complex state with shape
                ``(...,) (self.K + 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        U_real = u[..., :self.K + 1]
        U_imag = zeros_like(U_real, device=next(self.parameters()).device.type)
        U_imag[..., 1:] = u[..., self.K + 1:]
        return U_real + 1j * U_imag

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 2)]`.

        This method returns an I.C. where :math:`\\|\\vec{x}\\|_2
        \\sim U[[0, 2)]`. In particular, an initial condition is sampled in the
        following way: First, :math:`\\vec{u} \\sim U[\\mathcal{S}^{n-1}]`.
        Second, :math:`r\\sim U[[0, 2)]`. Finally, the sample :math:`ru` is
        returned.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(...,) + (2 * (self.K + 1) - 1,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        gaussian = randn((self.num_states,),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = 2.0 * rand((1,), device=next(self.parameters()).device.type)
        return radius * direction

    def _dealiased_irfft(self, U: Tensor) -> Tensor:
        U_pad = zeros(U.shape[:-1] + (self._K_prime + 1,),
                      dtype=U.dtype,
                      device=next(self.parameters()).device.type).squeeze(0)
        U_pad[..., :self.K + 1] = U[..., :self.K + 1]
        return irfft(U_pad, n=self._N_prime, norm='forward')

    def _dealiased_rfft(self, u: Tensor) -> Tensor:
        return rfft(u, norm='forward')[..., :self.K + 1]
