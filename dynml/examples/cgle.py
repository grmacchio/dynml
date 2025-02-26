"""Contain all code related to the complex Ginzburg-Landau equation.

This module contains all code related to the complex Ginzburg-Landau equation.
"""


# import built-in python-package code
# None
# import external python-package code
from scipy.special import roots_hermite  # type: ignore  # no stubs
from torch import abs, cumsum, diag, exp, eye, hstack, ones, pow, rand, randn
from torch import reshape, sqrt, tensor, Tensor, vstack, zeros, zeros_like
from torch.linalg import norm
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem


# export public code
__all__ = ['CGLE']


# TODO: Define
class CGLE(SemiLinearFirstOrderSystem):
    """Represent the discretized complex Ginzburg-Landau equation.

    This class represents the discretized complex Ginzburg-Landau equation.
    The complex Ginzburg-Landau equation (C.G.L.E.) is a partial
    differential equation (P.D.E.) of the form described in [1]:

    .. math::
        \\frac{\\partial q}{\\partial t}
        = \\left( -\\nu \\frac{\\partial}{\\partial x}
          + \\gamma \\frac{\\partial^2}{\\partial x^2}
          + \\mu(\\:\\cdot_{x}) \\right) q - a |q|^2 q,

    where :math:`q(t, x) \\in \\mathbb{C}`, :math:`a > 0`,
    :math:`t\\in\\mathbb{R}`,
    :math:`x \\in \\mathbb{R}`, :math:`U \\in \\mathbb{R}`,
    :math:`c_u \\in \\mathbb{R}`, :math:`c_d \\in \\mathbb{R}`,
    :math:`\\mu_0 \\in \\mathbb{R}`, :math:`\\mu_2\\in \\mathbb{R}`,
    :math:`\\nu = U + 2 i c_u`, :math:`\\gamma = 1 + i c_d`, and

    .. math::
        \\mu(x) = (\\mu_0 - c_u^2) + \\frac{1}{2} \\mu_2 x^2.

    Lets now transform the C.G.L.E. into a two-dimensional real P.D.E. by
    setting :math:`q = q_r + i q_i`:

    .. math::
        \\begin{align*}
            \\frac{\\partial}{\\partial t} \\begin{bmatrix} q_r \\\\ q_i
            \\end{bmatrix} &= \\begin{bmatrix}
            - U \\frac{\\partial}{\\partial x}
            + \\frac{\\partial^2}{\\partial x^2} & 2 c_u
            \\frac{\\partial}{\\partial x}
            - c_d \\frac{\\partial^2}{\\partial x^2}
            \\\\ -2 c_u \\frac{\\partial}{\\partial x} + c_d
            \\frac{\\partial^2}{\\partial x^2}
            & -U \\frac{\\partial}{\\partial x}
            + \\frac{\\partial^2}{\\partial x^2} \\end{bmatrix}
            \\begin{bmatrix} q_r \\\\ q_i \\end{bmatrix} + \\begin{bmatrix}
            [-a(q_r^2 + q_i^2) + \\mu]q_r \\\\ [-a(q_r^2 + q_i^2) + \\mu] q_i
            \\end{bmatrix}
        \\end{align*}

    Next, we scale the spacial domain by :math:`\\tilde{x} = bx`, where
    :math:`b = |\\mu_2 / (2 \\gamma)|^{1/4}`, which results in a
    two-dimensional real P.D.E. with respect to the functions
    :math:`\\tilde{q}_r = q_r(\\:\\cdot_t, \\cdot_{\\tilde{x}}/b)` and
    :math:`\\tilde{q}_i = q_i(\\:\\cdot_t, \\cdot_{\\tilde{x}}/b)`. The
    reason for doing this is described in [2, 3]. In order to obtain a
    first-order O.D.E. system, we orthogonally project this P.D.E. onto the
    space of Hermite interpolants where solutions
    :math:`\\tilde{q}_r` and :math:`\\tilde{q}_i` take the form,

    .. math::
        \\begin{align*}
            \\tilde{q} =
                \\begin{bmatrix}
                    \\tilde{q}_r
                    \\\\
                    \\tilde{q}_i
                \\end{bmatrix}
                =
                \\begin{bmatrix}
                    \\sum_{j=1}^N \\left[
                    \\frac{\\alpha(\\:\\cdot_{\\tilde{x}})}{\\alpha(\\tilde{x}_j)}
                    \\frac{H_N(\\:\\cdot_{\\tilde{x}})}{H'_N(\\tilde{x}_j)
                    (\\:\\cdot_{\\tilde{x}}-\\tilde{x}_j)}
                    \\right] q_r(\\:\\cdot_{t}, \\tilde{x}_j/b)
                    \\\\
                    \\sum_{j=1}^N \\left[
                    \\frac{\\alpha(\\:\\cdot_{\\tilde{x}})}{\\alpha(\\tilde{x}_j)}
                    \\frac{H_N(\\:\\tilde{x})}{H'_N(\\tilde{x}_j)
                    (\\:\\cdot_{\\tilde{x}}- \\tilde{x}_j)}
                    \\right] q_i(\\:\\cdot_{t}, \\tilde{x}_j/b)
                \\end{bmatrix}
                =
                H
                \\begin{bmatrix}
                    q_r(\\:\\cdot_{t}, \\tilde{x}_1/b)
                    \\\\
                    \\vdots
                    \\\\
                    q_r(\\:\\cdot_{t}, \\tilde{x}_N/b)
                    \\\\
                    q_i(\\:\\cdot_{t}, \\tilde{x}_1/b)
                    \\\\
                    \\vdots
                    \\\\
                    q_i(\\:\\cdot_{t}, \\tilde{x}_N/b)
                \\end{bmatrix}
                =
                H \\vec{x},
        \\end{align*}

    where the collocation points :math:`\\{\\tilde{x}_j\\}_{j=1}^N` are the
    roots of the Hermite polynomial of degree :math:`N` and
    :math:`\\alpha(\\tilde{x}) = e^{-\\tilde{x}^2/2}` [3]. By substituting
    this expression into the P.D.E. and projecting, we obtain a system of
    real-valued ordinary differential equations of the form,

    .. math::
        \\begin{align*}
            \\frac{d}{dt} \\vec{x} &= H^\\dagger L H \\vec{x}
            + H^\\dagger N( H\\vec{x}),
        \\end{align*}

    where :math:`H^\\dagger` is the pseudo-inverse of the Hermite interpolation
    matrix :math:`H`. The linear term :math:`H^\\dagger L H\\vec{x}` is
    computed using the methodology described in [3],

    .. math::
        \\begin{align*}
            H^\\dagger L H \\vec{x} =
            \\begin{bmatrix}
                -U b D_1 + b^2 D_2
                & 2 c_u b D_1 - c_d b^2 D_2
                \\\\
                -2 c_u b D_1 + c_d b^2 D_2
                & -U b D_1 + b^2 D_2
            \\end{bmatrix} \\vec{x}
        \\end{align*}

    where :math:`D_1` and :math:`D_2` are the first and second derivative
    matrices, respectively. The nonlinear term :math:`H^\\dagger N (H\\vec{x})`
    is approximated by applying the nonlinearity to the solution's values at
    the collocation points:

    .. math::
        \\begin{align*}
            H^\\dagger N (H\\vec{x}) \\approx
            \\begin{bmatrix}
                -a \\left[q_r(\\:\\cdot_{t}, \\tilde{x}_1/b)^2
                + q_i(\\:\\cdot_{t}, \\tilde{x}_1/b)^2\\right]
                q_r(\\:\\cdot_{t}, \\tilde{x}_1/b)
                \\\\
                \\vdots
                \\\\
                -a \\left[q_r(\\:\\cdot_{t}, \\tilde{x}_N/b)^2
                + q_i(\\:\\cdot_{t}, \\tilde{x}_N/b)^2\\right]
                q_r(\\:\\cdot_{t}, \\tilde{x}_N/b)
                \\\\
                -a \\left[q_r(\\:\\cdot_{t}, \\tilde{x}_1/b)^2
                + q_i(\\:\\cdot_{t}, \\tilde{x}_1/b)^2\\right]
                q_i(\\:\\cdot_{t}, \\tilde{x}_1/b)
                \\\\
                \\vdots
                \\\\
                -a \\left[q_r(\\:\\cdot_{t}, \\tilde{x}_N/b)^2
                + q_i(\\:\\cdot_{t}, \\tilde{x}_N/b)^2\\right]
                q_i(\\:\\cdot_{t}, \\tilde{x}_N/b)
            \\end{bmatrix}
            +
            \\begin{bmatrix}
                \\mu(\\tilde{x}_1/b) \\\\
                \\vdots \\\\
                \\mu(\\tilde{x}_N/b) \\\\
                \\mu(\\tilde{x}_1/b) \\\\
                \\vdots \\\\
                \\mu(\\tilde{x}_N/b)
            \\end{bmatrix} \\odot \\vec{x}.
        \\end{align*}

    Grouping linear terms together, we obtain the final system of first-order
    ordinary differential equations of the form,

    .. math::
        \\begin{align*}
            \\frac{d}{dt} \\vec{x} &= A \\vec{x} + F(\\vec{x}) = f(\\vec{x}).
        \\end{align*}

    The output equation is defined by

    .. math::
        \\begin{align*}
            \\vec{y}(\\:\\cdot_{t}) &= \\int_{-\\infty}^{\\infty}
            e^{-(\\:\\cdot_{\\tilde{x}}/b - x_B)^2/\\sigma^2}
            \\begin{bmatrix}
                q_r(\\:\\cdot_{t}, \\cdot_{\\tilde{x}}/b) \\\\
                q_i(\\:\\cdot_{t}, \\cdot_{\\tilde{x}}/b)
            \\end{bmatrix} \\frac{1}{b} \\, d\\tilde{x}
            \\\\
            &\\approx \\sum_{j=1}^N w_j
            e^{-(\\tilde{x}_j/b - x_B)^2/\\sigma^2}
            \\begin{bmatrix}
                q_r(\\:\\cdot_{t}, \\tilde{x}_j/b) \\\\
                q_i(\\:\\cdot_{t}, \\tilde{x}_j/b)
            \\end{bmatrix} \\frac{1}{b} = C \\vec{x},
        \\end{align*}

    where :math:`w_j` are defined using the trapzoidal rule and :math:`x_B`
    is "branch II" [2].

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``N`` (``int``): the number of Hermite interpolants
    |   ``U`` (``float``): the value of the parameter :math:`U`
    |   ``cu`` (``float``): the value of the parameter :math:`c_u`
    |   ``cd`` (``float``): the value of the parameter :math:`c_d`
    |   ``mu0`` (``float``): the value of the parameter :math:`\\mu_0`
    |   ``mu2`` (``float``): the value of the parameter :math:`\\mu_2`
    |   ``a`` (``float``): the value of the parameter :math:`a`
    |   ``A`` (``Tensor``): the matrix :math:`A`
    |   ``num_states`` (``int``): the number of states
    |   ``C`` (``Tensor``): the matrix :math:`C`
    |   ``num_outputs`` (``int``): the number of outputs

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``nonlinear()``: return :math:`F(\\vec{x})`
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``rhs()``: return :math:`f(\\vec{x})`
    |   ``gen_ic()``: return an I.C. where
            :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 3)]`
    |   ``output()``: return :math:`C\\vec{x}`

    | **References**
    |   [1] Ilak, Miloš, et al. "Model reduction of the nonlinear complex
            Ginzburg–Landau equation." SIAM Journal on Applied Dynamical
            Systems 9.4 (2010): 1284-1302.
    |   [2] Bagheri, Shervin, et al. "Input-output analysis and control design
            applied to a linear model of spatially developing flows." (2009):
            020803.
    |   [3] Weideman, J. Andre, and Satish C. Reddy. "A MATLAB differentiation
            matrix suite." ACM transactions on mathematical software (TOMS)
            26.4 (2000): 465-519.
    """

    @property
    def A(self) -> Tensor:
        return self._A

    @property
    def num_states(self) -> int:
        return 2 * self.N

    def nonlinear(self, x: Tensor) -> Tensor:
        """Return :math:`F(\\vec{x})`.

        This method returns :math:`F(\\vec{x})`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) +(2 * self.N,)``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape
                ``(...,) +(2 * self.N,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Ilak, Miloš, et al. "Model reduction of the nonlinear complex
                Ginzburg–Landau equation." SIAM Journal on Applied Dynamical
                Systems 9.4 (2010): 1284-1302.
        |   [2] Bagheri, Shervin, et al. "Input-output analysis and control
                design applied to a linear model of spatially developing
                flows." (2009): 020803.
        """
        # separate the real and imaginary parts
        xr = x[..., :self.N]
        xi = x[..., self.N:]
        # compute the nonlinear term
        F = zeros_like(x)
        x_sq = (xr**2 + xi**2)
        F[..., :self.N] = x_sq * xr
        F[..., self.N:] = x_sq * xi
        return -self.a * F

    def __init__(self, N: int = 220, U: float = 2.0,  # noqa: C901
                 cu: float = 0.2, cd: float = -1.0, mu0: float = 0.41,
                 mu2: float = -0.01, a: float = 0.1, sigma: float = 1.6):
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``N`` (``int``): the number of Hermite interpolants
        |   ``U`` (``float``): the value of the parameter :math:`U`
        |   ``cu`` (``float``): the value of the parameter :math:`c_u`
        |   ``cd`` (``float``): the value of the parameter :math:`c_d`
        |   ``mu0`` (``float``): the value of the parameter :math:`\\mu_0`
        |   ``mu2`` (``float``): the value of the parameter :math:`\\mu_2`
        |   ``a`` (``float``): the value of the parameter :math:`a`
        |   ``sigma`` (``float``): the value of the parameter :math:`\\sigma`

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   [1] Ilak, Miloš, et al. "Model reduction of the nonlinear complex
                Ginzburg–Landau equation." SIAM Journal on Applied Dynamical
                Systems 9.4 (2010): 1284-1302.
        |   [2] Bagheri, Shervin, et al. "Input-output analysis and control
                design applied to a linear model of spatially developing
                flows." (2009): 020803.
        |   [3] Weideman, J. Andre, and Satish C. Reddy. "A MATLAB
                differentiation matrix suite." ACM transactions on mathematical
                software (TOMS) 26.4 (2000): 465-519.
        """
        # initialize the superclass and model parameters
        super().__init__()
        self.N = N
        self.U = U
        self.cu = cu
        self.cd = cd
        self.mu0 = mu0
        self.mu2 = mu2
        self.a = a
        # calculate the collocation points and weights on page 471 of [3]
        self.x_tilde = tensor(roots_hermite(self.N)[0])
        alpha = exp(-self.x_tilde**2 / 2.0)
        # define the C matrix defined on page 473 of [3]
        c = ones(self.N)
        for j in range(self.N):
            for m in range(self.N):
                if m != j:
                    c[j] = c[j] * (self.x_tilde[j] - self.x_tilde[m])
            c[j] = alpha[j] * c[j]
        C = zeros((self.N, self.N))
        for k in range(self.N):
            for j in range(self.N):
                C[k, j] = c[k] / c[j]
        # define the B matrix defined on page 473 and Equation 18 of [3]
        B = zeros((2, self.N))
        B[0, :] = -self.x_tilde * 1
        B[1, :] = -self.x_tilde * B[0, :] - 1 * 1
        # define the Z matrix on page 473 of [3]
        Z = zeros((self.N, self.N))
        for k in range(self.N):
            for j in range(self.N):
                if j != k:
                    Z[k, j] = 1.0 / (self.x_tilde[k] - self.x_tilde[j])
        # define the X matrix on page 473 of [3]
        X = zeros((self.N - 1, self.N))
        for k in range(self.N - 1):
            for j in range(self.N):
                if j <= k:
                    X[k, j] = 1.0 / (self.x_tilde[j] - self.x_tilde[k + 1])
                else:
                    X[k, j] = 1.0 / (self.x_tilde[j] - self.x_tilde[k])
        # construct the diff. matrices using Table II on page 474 of [3]
        Y = ones((self.N - 1, self.N))
        D = eye(self.N)
        D_mats = list()
        for k in range(2):
            ell = k + 1
            Y = cumsum(vstack([B[ell - 1, :].reshape(1, -1),
                              ell * Y[:self.N - 1, :] * X]), dim=0)
            D_repmat = reshape(diag(D), (-1, 1)) * ones((self.N,
                                                         self.N))
            D = ell * Z * (C * D_repmat - D)
            for i in range(self.N):
                D[i, i] = Y[self.N - 1, i]
            D_mats.append(D)
        # calculate the domain scaling factor, b, suggested in [2, 3].
        self.b = pow(abs(self.mu2 / (2 * tensor(1.0 + 1j * self.cd))),
                     tensor(0.25))
        # construct the A matrix
        def mu(x: Tensor) -> Tensor:  # noqa: E306
            return (self.mu0 - self.cu**2) + 0.5 * self.mu2 * x**2
        mu_diag = diag(mu(self.x_tilde / self.b))
        Arr = -U * self.b * D_mats[0] + self.b ** 2 * D_mats[1] + mu_diag
        Ari = 2 * cu * self.b * D_mats[0] - cd * self.b ** 2 * D_mats[1]
        Air = -2 * cu * self.b * D_mats[0] + cd * self.b ** 2 * D_mats[1]
        Aii = -U * self.b * D_mats[0] + self.b ** 2 * D_mats[1] + mu_diag
        self._A = Parameter(vstack([hstack([Arr, Ari]), hstack([Air, Aii])]),
                            requires_grad=False)
        # construct the output matrix
        x_B = sqrt(tensor(-2 * (self.mu0 - self.cu ** 2) / self.mu2))
        ker_obs = exp(-((self.x_tilde / self.b - x_B) / sigma)**2)
        C = zeros((2, self.num_states))
        weights = zeros(self.N)
        for i in range(self.N):
            if i == 0:
                weights[i] = 0.5 * (self.x_tilde[i + 1] - self.x_tilde[i])
            elif i == self.N - 1:
                weights[i] = 0.5 * (self.x_tilde[i] - self.x_tilde[i - 1])
            else:
                weights[i] = 0.5 * (self.x_tilde[i + 1] - self.x_tilde[i - 1])
        C[0, :self.N] = ker_obs * weights / self.b
        C[1, self.N:] = ker_obs * weights / self.b
        self.C = Parameter(C, requires_grad=False)
        self.num_outputs = 2

    def gen_ic(self) -> Tensor:
        """Return an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim U[[0, 3)]`.

        This method returns an I.C. where :math:`\\|\\vec{x}\\|_2 \\sim
        U[[0, 3)]`. In particular, an initial condition is sampled in the
        following way: First, :math:`\\vec{u} \\sim U[\\mathcal{S}^{n-1}]`.
        Second, :math:`r\\sim U[[0, 3)]`. Finally, the sample :math:`ru` is
        returned.


        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape
                ``(2 * self.N,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the initial condition where ||x||_2 ~ U[[0, 3)]
        gaussian = randn((self.num_states,),
                         device=next(self.parameters()).device.type)
        direction = gaussian / norm(gaussian)
        radius = 3.0 * rand((1,), device=next(self.parameters()).device.type)
        return radius * direction

    def output(self, x: Tensor) -> Tensor:
        """Return :math:`C\\vec{x}`.

        This method returns :math:`C\\vec{x}`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) +(2 * self.N,)``

        | **Returns**
        |   ``Tensor``: the output with shape ``(...,) + (2,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Ilak, Miloš, et al. "Model reduction of the nonlinear complex
                Ginzburg–Landau equation." SIAM Journal on Applied Dynamical
                Systems 9.4 (2010): 1284-1302.
        |   [2] Bagheri, Shervin, et al. "Input-output analysis and control
                design applied to a linear model of spatially developing
                flows." (2009): 020803.
        """
        # return the output
        return x @ self.C.T
