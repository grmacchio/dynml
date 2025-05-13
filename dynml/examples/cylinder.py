"""Contain all code related to a system describing a 2D cylinder wake.

This module contains all code related to a system describing a 2D cylinder
wake.
"""


# import built-in python-package code
from typing import Tuple
from math import sqrt
# import external python-package code
from torch import inf, max, rand, Tensor, where, zeros
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.discrete.system import DiscreteSystem


# export public code
__all__ = ['Cylinder']


# argument parser choice dictionaries
class Cylinder(DiscreteSystem):
    """Represent a system describing a 2D cylinder wake.

    This class represents a dynamical system describing a 2D cylinder wake. In
    particular, this model is a finite-difference model of the incompressible
    Navier-Stokes equations on a square domain.
    
    Let :math:`\\mathcal{C} \\subseteq [0, 1]^2` be a closed ball of radius
    :math:`R` positioned at :math:`s_c = (1/5, 1/2)`. The spacial
    domain is then :math:`\\Omega = [0, 1]^2 - \\mathrm{Int}\:\\mathcal{C}`. On
    this domain, the incompressible Navier-Stokes equations are then given by

    .. math::
        \\begin{align*}
            \\frac{\\partial u}{\\partial t} + \\mathrm{D}_s \\: u \\: u
            &= - \\frac{1}{\\rho} \\nabla_s \\: p + \\nu (\\nabla_s \\cdot
            \\nabla_s) u + g \\\\
            \\nabla_s \\cdot u &= 0.
        \\end{align*}
    
    The boundary conditions for the flow field are given by the following.

    .. math::
        \\begin{align*}
            (\\text{BC1}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 0 &\\implies u(t, s) = (U_\\infty, 0) \\\\
            &\\implies \\partial u_2 / \\partial s_1 = 0 \\\\
            (\\text{BC2}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 1 &\\implies u(t, s) = (U_\\infty, 0) \\\\
            &\\implies \\partial u_1 / \\partial s_2 = 0 \\\\
            (\\text{BC3}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 0 &\\implies u(t, s) = (U_\\infty, 0) \\\\
            &\\implies \\partial u_1 / \\partial s_2 = 0 \\\\
            (\\text{BC4}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 1 &\\implies \\partial u / \\partial s_1 = 0 \\\\
            (\\text{BC5}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall x \\in \\partial \\mathcal{C}, \\: u(t, s) = \\: &0
        \\end{align*}

    Now that we have defined the problem, lets reformulate it in terms of the
    :math:`\\psi`-:math:`\\omega` formulation. Because the flow is
    :math:`2`-dimensional, satisfies continuity, and :math:`\\Omega`
    only has one hole with no net flux inwards or outwards, we can say there
    exists a stream function :math:`\\psi` such that

    .. math::
        \\begin{align*}
            u = \\frac{\\partial \\psi}{\\partial s_2} e_{1}
            - \\frac{\\partial \\psi}{\\partial s_1} e_{2},
        \\end{align*}

    where :math:`e_1` and :math:`e_2` are the unit vectors in the :math:`s_1`
    and :math:`s_2` directions, respectively. If we let
    :math:`\\omega = \\mathrm{curl}_s \\: u`, then we can relate :math:`\\psi`
    to :math:`\\omega` by the following equation

    .. math::
        \\begin{align*}
            \\omega = -(\\nabla_s \\cdot \\nabla_s) \\psi.
        \\end{align*}

    Next, if we remember that :math:`\\mathrm{curl}_s \\: \\nabla_s p = 0` and
    :math:`\\mathrm{curl}_s \\: g = 0`, we can apply the curl operator to the
    momentum equation and , after some algebraic manipulation, we can rewrite
    the incompressible Navier-Stokes equations as

    .. math::
        \\begin{align*}
                \\frac{\\partial \\omega}{\\partial t}
                + \\nabla_s \\cdot \\left(\\omega \\: u\\right)
                &= \\nu (\\nabla_s \\cdot \\nabla_s) \\omega \\\\
                \\omega &= -(\\nabla_s \\cdot \\nabla_s) \\psi.
        \\end{align*}

    Next, we need to derive equivalent boundary conditions in this new
    formulation. The boundary conditions are converted into the following.

    .. math::
        \\begin{align*}
            (\\text{BC1}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 0 &\\implies \\psi = U_\\infty (s_2 - 1/2) \\\\
            &\\implies \\omega = 0 \\\\
            (\\text{BC2}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 1 &\\implies \\psi = U_\\infty / 2 \\\\
            &\\implies \\omega = 0 \\\\
            (\\text{BC3}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 0 &\\implies \\psi = -U_\\infty / 2 \\\\
            &\\implies \\omega = 0 \\\\
            (\\text{BC4}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 1 &\\implies \\partial^2 \\omega / \\partial s_1^2
            = \\partial^2\\psi / \\partial s_1^2 = 0 \\\\
            (\\text{BC5}) \\qquad \\forall t \in [0, \\infty), \\:
            \\forall s \\in \\partial \\mathcal{C},
            \\: \\psi = \\: 0 \\quad \\: \\: &
        \\end{align*}

    In fact, :math:`\\psi` can take any constant value on the boundary of the
    cylinder. Here, we have chosen :math:`\\psi = 0` for convenience. After
    determining :math:`\\text{BC5}`, we determine that
    :math:`\psi(t, 0, 1/2) = 0` because it must match the value of the stream
    function at the stagnation point. This allows us to determine the
    functional form of :math:`\\psi` in
    :math:`\\text{BC1}`, :math:`\\text{BC2}`, and :math:`\\text{BC3}`.

    Lets now go through the discretization process. In this model, we set the
    grid space domain to :math:`G = [1 : N_1] \\times [1 : 2 N_2 + 1]`, where
    the odd number of points in the :math:`s_2`-direction is to include the
    stagnation streamline. Let :math:`\\Delta s_1 = 1 / (N_1 - 1)` and
    :math:`\\Delta s_2 = 1 / (2 N_2)`. 

    Let :math:`x = (\\omega_{i, j})_{(i, j) \\in G - \\partial G}` be the
    state. Of course, one could cut out the interior cylinder points since
    they are determined by the exterior values of :math:`x_{i, j}`. Here, for
    computational simplicity, we will keep the boundary points of the cylinder
    in the state. Let :math:`x` be a state satisfying the boundary conditions.
    Please see ``gen_ic()`` for an example on how to generate such a state.

    Lets now discretize the equations. Given a state :math:`x_{i, j}` we
    generate the vorticity :math:`\\omega_{i, j}` by (1) setting the left, top,
    and bottom boundary conditions to zero and (2) determining the right
    boundary by doing a extrapolation with a left-weighted second-order kernel:

    .. math::
        \\begin{align*}
            \\frac{\\partial^2 \omega}{ \\partial^2 s_1} \\approx 
            \\frac{2 \\omega_{2 N_1 + 1, j} - 5 \\omega_{2 N_1, j}
            + 4 \\omega_{2 N_1 - 1, j}
            - \\omega_{2 N_1 - 2, j}}{\\Delta s_1^3} \\\\
            \\omega_{2 N_1 + 1, j} = \\frac{1}{2}\\left(
                5 \\omega_{2 N_1, j} - 4 \\omega_{2 N_1 - 1, j}
                + \\omega_{2 N_1 - 2, j}
            \\right).
        \\end{align*}
    
    We now have the tensor :math:`\\omega_{i, j}`. From this, we solve the
    points points of :math:`\\psi_{i, j}` by solving the Poisson equation. This
    was done using the Jacobi method defined by

    .. math::
        \\begin{align*}
            \\psi_{i, j}^{(k+1)} = \\frac{1}{2 \\left(
                \\frac{1}{\\Delta s_1^2} + \\frac{1}{\\Delta s_2^2}
            \\right)} \\left(
                \\frac{\\psi_{i + 1, j}^{(k)}
                + \\psi_{i - 1}^{(k)}}{\\Delta s_1^2}
                + \\frac{\\psi_{i, j + 1}^{(k)}
                + \\psi_{i, j - 1}^{(k)}}{\\Delta s_2^2}
                + \\omega_{i, j}
            \\right).
        \\end{align*}
    
    for :math:`(i, j) \\in G - \\partial G`. The Jacobi method was used here
    because I could tensorize the operation rather than using a two ``for``
    loops in Gauss-Seidel. One then enforces the boundary point values by using
    :math:`\\text{BC1}` through :math:`\\text{BC5}`. The right boundary points
    are set using the same left-weighted second-order extrapolation as was done
    for vorticity. Here, we also set all points inside the cylinder to zero.
    Next, lets calculate the velocity field using the stream function:

    .. math::
        \\begin{align*}
            u_{1, i, j} = \\frac{\\psi_{i, j + 1} - \\psi_{i, j - 1}}{2
            \\Delta s_2}, \\quad
            v_{1, i, j} = - \\frac{\\psi_{i + 1, j} - \\psi_{i - 1, j}}{2
            \\Delta s_1}.
        \\end{align*}

    for :math:`(i, j) \\in G - \\partial G`. The velocity field on the boundary
    points is set by using :math:`\\text{BC1}` through :math:`\\text{BC5}`. The
    right boundary points are calculated using a left-weighted second-order
    approximation for :math:`\\partial \\psi / \\partial s_1`. Following this
    calculation of :math:`u_{i, j}` and :math:`v_{i, j}`, we can now calculate
    the update of the vorticity field: :math:`\\omega_{i, j}` on
    :math:`G - \\partial G`. Here, we use Euler's method to update in time, we
    use upwind differencing to calculate the advection term, and we use
    central differencing to calculate the diffusion term. The update is
    given explicitly by

    .. math::
        \\begin{align*}
            \\omega_{i, j} + \\Delta t \\left(
                - \\text{UppDiff}_{s_1} (u, \\omega)
                - \\text{UppDiff}_{s_2} (v, \\omega)
                + \\nu \\text{Lap} (\\omega)
            \\right)
        \\end{align*}

    where

    .. math::
        \\begin{align*}
            \\text{UppDiff}_{s_1} (u, \\omega) &=
            \\begin{cases}
                \\frac{u_{i, j} \\omega_{i, j} - u_{i - 1, j}
                \\omega_{i - 1, j}}{\\Delta s_1} & \\text{if } u_{i, j} > 0
                \\\\
                \\frac{u_{i + 1, j} \\omega_{i + 1, j}
                - u_{i, j} \\omega_{i, j}}{\\Delta s_1} & \\text{if }
                u_{i, j} < 0
            \\end{cases} \\\\
            \\text{UppDiff}_{s_2} (v, \\omega) &=
            \\begin{cases}
                \\frac{v_{i, j} \\omega_{i, j} - v_{i, j - 1}
                \\omega_{i, j - 1}}{\\Delta s_2} & \\text{if } v_{i, j} > 0 \\\\
                \\frac{v_{i, j + 1} \\omega_{i, j + 1}
                - v_{i, j} \\omega_{i, j}}{\\Delta s_2} & \\text{if }
                v_{i, j} < 0
            \\end{cases} \\\\
            \\text{Lap} (\\omega) &= \\frac{\\omega_{i + 1, j}
            - 2 \\omega_{i, j} + \\omega_{i - 1, j}}{\\Delta s_1^2}
            + \\frac{\\omega_{i, j + 1} - 2 \\omega_{i, j}
            + \\omega_{i, j - 1}}{\\Delta s_2^2}.
        \\end{align*}

    This determines all values of the next state :math:`\\Phi_{\\Delta t}(x)`.

    Finally, this class' initialization method enforces two conditions: First,
    that both grid spacings satisfy the condition for good diffusion modeling:

    .. math::
        \\begin{align*}
            Re_{\\Delta s_1} = \\frac{U_\\infty \\Delta s_1}{\\nu} < 10,
            \\quad
            Re_{\\Delta s_2} = \\frac{U_\\infty \\Delta s_2}{\\nu} < 10.
        \\end{align*}

    Second, the time step satisfies the Courant-Friedrichs-Lewy condition:

    .. math::
        \\begin{align*}
            \\Delta t < \\frac{\\min(\\Delta s_1, \\Delta s_2)}{U_{\\max}}.
        \\end{align*}

    where for cylinder flow we know :math:`U_{\\max} = 2 U_\\infty`.
            
    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``m`` (``int``): the number of grid points in the :math:`s_2`-direction
    |   ``n`` (``int``): the number of grid points in the :math:`s_1`-direction
    |   ``nu`` (``float``): the value of the parameter :math:`\\nu`
    |   ``R`` (``float``): the value of the parameter :math:`R`
    |   ``U_inf`` (``float``): the value of the parameter :math:`U_\\infty`
    |   ``ds2`` (``float``): the value of the parameter :math:`\\Delta s_2`
    |   ``ds1`` (``float``): the value of the parameter :math:`\\Delta s_1`
    |   ``dt`` (``float``): the value of the parameter :math:`\\Delta t`
    |   ``ep`` (``float``): the value of the parameter :math:`\\epsilon`
    |   ``s_c`` (``Tuple[float, float]``): the value of the parameter
            :math:`s_c`
    |   ``field`` (``str``): ``R`` for real numbers
    |   ``dims_state`` (``int``): the state dimensions
    |   ``mask`` (``Tensor``): the mask for grid point delineation

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``__init__()``: initialize the superclass and model parameters
    |   ``map()``: return :math:`\\Phi_{\\Delta t}(x)`
    |   ``gen_ic()``: return a random initial condition

    | **References**
    |   None
    """

    @property
    def field(self) -> str:
        return 'R'

    @property
    def dims_state(self) -> Tuple[int, ...]:
        return self._dims_state

    def map(self, x: Tensor) -> Tensor:
        """Return :math:`\\Phi_{\\Delta t}(x)`.

        This method returns :math:`\\Phi_{\\Delta t}(x)`.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape ``(...,) + dims_state``

        | **Returns**
        |   ``Tensor``: the nonlinear term with shape ``(...,) + dims_state``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # compute the nonlinear term
        omega = self.x_to_omega(x)
        u, v = self._psi_to_vel(self.psi_prev)
        output = zeros((self.m, self.n))
        omega_i_j = omega[..., 1:self.m-1, 1:self.n-1]
        omega_ip1_j = omega[..., 1:self.m-1, 2:self.n]
        omega_im1_j = omega[..., 1:self.m-1, 0:self.n-2]
        omega_i_jp1 = omega[..., 0:self.m-2, 1:self.n-1]
        omega_i_jm1 = omega[..., 2:self.m, 1:self.n-1]
        laplacian1 = (omega_ip1_j - 2 * omega_i_j + omega_im1_j) / self.ds1**2
        laplacian2 = (omega_i_jp1 - 2 * omega_i_j + omega_i_jm1) / self.ds2**2
        laplacian = laplacian1 + laplacian2
        u_i_j = u[..., 1:self.m-1, 1:self.n-1]
        u_ip1_j = u[..., 1:self.m-1, 2:self.n]
        u_im1_j = u[..., 1:self.m-1, 0:self.n-2]
        Dwu_neg = (u_ip1_j * omega_ip1_j - u_i_j * omega_i_j)
        Dwu_pos = (u_i_j * omega_i_j - u_im1_j * omega_im1_j)
        Dwu = where(u[1:self.m-1, 1:self.n-1] > 0.0, Dwu_pos, Dwu_neg)
        v_i_j = v[..., 1:self.m-1, 1:self.n-1]
        v_i_jp1 = v[..., 0:self.m-2, 1:self.n-1]
        v_i_jm1 = v[..., 2:self.m, 1:self.n-1]
        Dwv_neg = (v_i_jp1 * omega_i_jp1 - v_i_j * omega_i_j)
        Dwv_pos = (v_i_j * omega_i_j - v_i_jm1 * omega_i_jm1)
        Dwv = where(v[1:self.m-1, 1:self.n-1] > 0.0, Dwv_pos, Dwv_neg)
        output[..., 1:self.m-1, 1:self.n-1] = (omega_i_j
                                               + self.dt * (- Dwu / self.ds1
                                                            - Dwv / self.ds2
                                                            + (self.nu
                                                               * laplacian)))
        output = where(self.mask == 0, output, 0.0)
        output[..., 1:self.m-1, -1] = 0.5 * (5 * output[..., 1:self.m-1, -2]
                                             - 4 * output[..., 1:self.m-1, -3]
                                             + output[..., 1:self.m-1, -4])
        self.psi_prev = self._poisson(output)
        return self.omega_to_x(self._add_bd_omega(output, self.psi_prev))


    def __init__(self, N_1: int = 600, N_2: int = 300, nu: float = 1.57e-5,
                 R: float = 0.05, Re: float = 200.0,
                 h_rf: float = 2.0/5.0, dt_rf: float = 0.5,
                 ep: float = 1e-7) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``N_1`` (``float``): the value of the parameter :math:`N_1` with
                default value of ``600``
        |   ``N_2`` (``float``): the value of the parameter :math:`N_2`
                with default value of ``300``
        |   ``nu`` (``float``): the value of the parameter :math:`\\nu` with
                default value of ``1.57e-5``
        |   ``R`` (``float``): the value of the parameter :math:`R` with
                default value of ``0.05``
        |   ``Re`` (``float``): the value of the parameter :math:`Re` with
                default value of ``200.0``
        |   ``h_rf`` (``float``): the value of the parameter
                :math:`h_{\\text{rf}}` with default value of ``2.0/5.0``
        |   ``dt_rf`` (``float``): the value of the parameter
                :math:`dt_{\\text{rf}}` with default value of ``0.5``
        |   ``ep`` (``float``): the value of the parameter :math:`\\epsilon`
                with default value of ``1e-7``

        | **Returns**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__()
        # set the discrete space parameters
        self.N_1 = N_1
        self.N_2 = N_2
        self.m = 2 * self.N_2 + 1
        self.n = self.N_1
        self._dims_state = (self.m - 2, self.n - 2)
        self.nu = nu
        # set the continuous space parameters
        self.R = R
        D = 2 * self.R
        self.U_inf = Re * nu / D
        self.ds2 = 1 / (2 * N_2)
        self.ds1 = 1 / (N_1 - 1)
        h = 10 * self.nu / self.U_inf * h_rf
        if (self.ds2 > h) or (self.ds1 > h):
            raise ValueError('Grid space must be smaller than 10 * self.nu / '
                             + 'self.U_inf * h_rf')
        self.dt = (min(self.ds2, self.ds1) / (2 * self.U_inf)) * dt_rf
        self.ep = ep
        self.s_c = (1/4, 1/2)
        # set the mask
        self.mask = zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                if sqrt((i * self.ds2 - self.s_c[1])**2
                        + (j * self.ds1 - self.s_c[0])**2) > R:
                    self.mask[i, j] = 0.0
                else:
                    self.mask[i, j] = 1.0
        for i in range(self.m):
            for j in range(self.n):
                if self.mask[i, j] == 1.0:
                    cond_1 = (self.mask[i - 1, j] == 0.0)
                    cond_2 = (self.mask[i + 1, j] == 0.0)
                    cond_3 = (self.mask[i, j - 1] == 0.0)
                    cond_4 = (self.mask[i, j + 1] == 0.0)
                    if cond_1 or cond_2 or cond_3 or cond_4:
                        self.mask[i, j] = 2.0
        # set up dummy parameter
        self._dummy = Parameter(zeros((1,)), requires_grad=False)

    def gen_ic(self) -> Tensor:
        """Return a random initial condition.

        This method returns a random initial condition. This is done by
        generating a random vorticity field and then solving the Poisson
        equation to get the stream function. After the boundary vorticity is
        added the initial condition is returned.

        | **Args**
        |   None

        | **Returns**
        |   ``Tensor``: the initial condition with shape ``self.dims_state``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # add boundary conditions to psi initial guess
        self.psi_prev = zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                if sqrt((i * self.ds2 - self.s_c[1])**2
                        + (j * self.ds1 - self.s_c[0])**2) > self.R:
                    self.psi_prev[i, j] = (- self.U_inf
                                           * (i * self.ds2 - self.s_c[1]))
                else:
                    self.psi_prev[i, j] = 0.0
        self.psi_prev[1:self.m-1, -1] = 0.5 * (5 * self.psi_prev[1:self.m-1, -2]
                                               - 4 * self.psi_prev[1:self.m-1, -3]
                                               + self.psi_prev[1:self.m-1, -4])
        # return the initial condition with added boundary vorticity
        omega = zeros((self.m, self.n))
        omega[1:self.m-1, 1:self.n-1] = 2 * rand(self.m - 2, self.n - 2) - 1
        omega[1:self.m-1, -1] = 0.5 * (5 * omega[1:self.m-1, -2]
                                        - 4 * omega[1:self.m-1, -3]
                                        + omega[1:self.m-1, -4])
        omega_bd = self._add_bd_omega(omega, self._poisson(omega))
        self.psi_prev = self._poisson(omega_bd)
        return self.omega_to_x(omega_bd)

    def x_to_omega(self, x: Tensor) -> Tensor:
        """Return the vorticity from the state.

        This method returns the vorticity from the state.

        | **Args**
        |   ``x`` (``Tensor``): the state with shape
                ``(...,) + self.dims_state``

        | **Returns**
        |   ``Tensor``: the vorticity with shape ``(...,) + (self.m, self.n)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        omega = zeros(x.shape[:-2] + (self.m, self.n))
        omega[..., 1:self.m-1, 1:self.n-1] = x[..., :]
        omega[..., 1:self.m-1, -1] = 0.5 * (5 * x[..., -1]
                                            - 4 * x[..., -2]
                                            + x[..., -3])
        return omega
    
    def omega_to_x(self, omega: Tensor) -> Tensor:
        """Return the state from the vorticity.

        This method returns the state from the vorticity.

        | **Args**
        |   ``omega`` (``Tensor``): the vorticity with shape
                ``(...,) + (self.m, self.n)``

        | **Returns**
        |   ``Tensor``: the state with shape ``(...,) + self.dims_state``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        x = zeros(omega.shape[:-1] + (self.m - 2, self.n - 2))
        x = omega[..., 1:self.m-1, 1:self.n-1]
        return x

    def _add_bd_omega(self, omega: Tensor, psi: Tensor) -> Tensor:
        omega_bd = zeros((self.m, self.n))
        term1 = (psi[1:self.m-1, 2:self.n]
                 - 2 * psi[1:self.m-1, 1:self.n-1]
                 + psi[1:self.m-1, 0:self.n-2]) / self.ds1**2
        term2 = (psi[2:self.m, 1:self.n-1]
                 - 2 * psi[1:self.m-1, 1:self.n-1]
                 + psi[0:self.m-2, 1:self.n-1]) / self.ds2**2
        omega_bd[1:self.m-1, 1:self.n-1] = - (term1 + term2)
        return where(self.mask == 2, omega_bd, omega)

    def _poisson(self, omega: Tensor) -> Tensor:
        psi_prev = self.psi_prev.clone()
        psi = self.psi_prev.clone()
        error = inf
        mult = 1 / (2 * (1 / self.ds1**2 + 1 / self.ds2**2))
        while error > self.ep:
            term1 = (psi_prev[..., 1:self.m-1, 2:self.n]
                     + psi_prev[..., 1:self.m-1, 0:self.n-2]) / self.ds1**2
            term2 = (psi_prev[..., 2:self.m, 1:self.n-1]
                     + psi_prev[..., 0:self.m-2, 1:self.n-1]) / self.ds2**2
            term3 = omega[..., 1:self.m-1, 1:self.n-1]
            psi[..., 1:self.m-1, 1:self.n-1] = mult * (term1 + term2 + term3)
            psi = where(self.mask == 0, psi, 0.0)
            error = max(abs(psi - psi_prev))
            psi_prev = psi.clone()
        return psi
    
    def _psi_to_vel(self, psi: Tensor) -> Tensor:
        u = zeros((self.m, self.n))
        u[0, :] = self.U_inf
        u[self.m - 1, :] = self.U_inf
        u[:, 0] = self.U_inf
        u[1:self.m-1, 1:] = (psi[0:self.m-2, 1:]
                             - psi[2:self.m, 1:]) / (2 * self.ds2)
        u = where(self.mask == 0, u, 0.0)
        v = zeros((self.m, self.n))
        v[0, :] = 0.0
        v[self.m - 1, :] = 0.0
        v[:, 0] = 0.0
        v[1:-1, 1:self.n-1] = (psi[1:-1, 2:self.n]
                               - psi[1:-1, 0:self.n-2]) / (2 * self.ds1)
        v[1:-1, self.n-1] = (3 * psi[1:-1, self.n-1]
                             - 4 * psi[1:-1, self.n-2]
                             + psi[1:-1, self.n-3]) / self.ds1**2
        v = -1 * where(self.mask == 0, v, 0.0)
        return u, v
