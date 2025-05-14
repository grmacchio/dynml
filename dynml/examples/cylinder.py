"""Contain all code related to a system describing a 2D cylinder wake.

This module contains all code related to a system describing a 2D cylinder
wake.
"""


# import built-in python-package code
from typing import Tuple
from math import sqrt
# import external python-package code
from torch import inf, max, no_grad, stack, Tensor, where, zeros
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

    Let :math:`\\mathcal{C} \\subseteq [0, L_1] \\times [0, 2 L_2]` be a closed
    ball of radius :math:`R` positioned at :math:`s_c = (s_{c,1}, L_2)`. The
    spacial domain is then
    :math:`\\Omega = [0, L_1] \\times [0, 2 L_2] - \\text{Int}\\:\\mathcal{C}`.
    If On this domain, the incompressible Navier-Stokes equations are then
    given by

    .. math::
        \\begin{align*}
            \\frac{\\partial u}{\\partial t} + \\mathrm{D}_s \\: u \\: u
            &= - \\frac{1}{\\rho} \\nabla_s \\: p + \\nu (\\nabla_s \\cdot
            \\nabla_s) u + g \\\\
            \\nabla_s \\cdot u &= 0.
        \\end{align*}

    There are five boundary conditions. The first condition enforces the
    velocity to be :math:`(U_\\infty, 0)` at the inlet with
    :math:`\\partial u_2 / \\partial s_1 = 0`. As we will see later, the
    aforementioned differential constraint corresponds to zero vorticity
    at the inlet. The second and third conditions enforce the velocity to be
    :math:`(U_\\infty, 0)` at the top and bottom of the domain with
    :math:`\\partial u_1 / \\partial s_2 = 0`. Similarly, the differential
    constraint corresponds to zero vorticity at the top and bottom of the
    domain. The fourth boundary condition comes from the following modeling
    assumption at the outlet: :math:`u \\approx (U_\\infty, 0)` at the outlet
    and :math:`\\partial u / \\partial t \\approx 0`. This is a reasonable
    assumption if the domain is large enough. The fifth condition enforces the
    velocity to be :math:`0` on the boundary of the cylinder. This is a no-slip
    condition. The boundary conditions are delineated as follows:

    .. math::
        \\begin{align*}
            &(\\text{BC1a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 2 L_2], \\:
            s_1 = 0 \\implies u = (U_\\infty, 0) \\\\
            &(\\text{BC1b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 2 L_2], \\:
            s_1 = 0  \\implies \\partial u_2 / \\partial s_1 = 0 \\\\
            &(\\text{BC2a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, L_1], \\:
            s_2 = 1 \\implies u = (U_\\infty, 0) \\\\
            &(\\text{BC2b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, L_1], \\:
            s_2 = 1 \\implies \\partial u_1 / \\partial s_2 = 0 \\\\
            &(\\text{BC3a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, L_1], \\:
            s_2 = 0 \\implies u = (U_\\infty, 0) \\\\
            &(\\text{BC3b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, L_1], \\:
            s_2 = 0 \\implies \\partial u_1 / \\partial s_2 = 0 \\\\
            &(\\text{BC4a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 2 L_2], \\:
            s_1 = 1 \\implies u \\approx (U_\\infty, 0) \\\\
            &(\\text{BC4b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 2 L_2], \\:
            \\implies \\partial u / \\partial t \\approx 0 \\\\
            &(\\text{BC5a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall x \\in \\partial \\mathcal{C}, \\: u(t, s) = \\: &0.
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
    formulation. Lets start by examining the boundary condition on
    the cylinder surface. Because :math:`u=0` on the cylinder surface, the
    partial derivatives of the stream function are zero. This means that
    :math:`\\psi` is constant on the cylinder surface. Here, we have chosen
    :math:`\\psi = 0` for convenience. Consequently, we are able to capture the
    stagnation stream line by letting :math:`\\psi(t, 0, 1/2) = 0`. Using this
    information, one can determine the first three boundary conditions by
    integration and the definition of of :math:`\\omega`. The fourth boundary
    condition is chosen by plugging the modeling assumptions the momentum
    equation: :math:`U_\\infty \\partial \\omega / \\partial s_1 \\approx \\nu
    (\\nabla_s \\cdot \\nabla_s) \\omega`. Another modeling assumption was to
    set :math:`\\partial \\psi / \\partial s_1 \\approx 0`. The fifth boundary
    condition is the no-slip condition on the cylinder surface. We know that
    :math:`\\psi` is constant on the cylinder surface; however, we still need
    to determine the values of :math:`\\omega` on :math:`\\partial
    \\mathcal{C}`. As we will see later, when we discretize the equation near
    the boundary of the cylinder, we will use the boundary conditions
    :math:`\\partial \\psi / \\partial s_1 = \\partial \\psi /
    \\partial s_2 = 0` to determine these boundary values of :math:`\\omega`.
    The boundary conditions are delineated as follows:

    .. math::
        \\begin{align*}
            &(\\text{BC1a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 0 \\implies \\psi = U_\\infty (s_2 - 1/2) \\\\
            &(\\text{BC1b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 0 \\implies \\omega = 0 \\\\
            &(\\text{BC2a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 1 \\implies \\psi = U_\\infty / 2 \\\\
            &(\\text{BC2b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 1 \\implies \\omega = 0 \\\\
            &(\\text{BC3a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 0 \\implies \\psi = -U_\\infty / 2 \\\\
            &(\\text{BC3b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_1 \\in [0, 1], \\:
            s_2 = 0 \\implies \\omega = 0 \\\\
            &(\\text{BC4a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 1 \\implies U_\\infty \\partial w / \\partial s_1 \\approx
            \\nu (\\nabla_s \\cdot \\nabla_s) \\omega \\\\
            &(\\text{BC4b}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s_2 \\in [0, 1], \\:
            s_1 = 1 \\implies \\partial \\psi / \\partial s_1 \\approx 0 \\\\
            &(\\text{BC5a}) \\qquad \\forall t \\in [0, \\infty), \\:
            \\forall s \\in \\partial \\mathcal{C},
            \\: (\\psi = 0) \\: \\land \\: (\\nabla_s \\psi = 0)
        \\end{align*}

    Lets now go through the discretization process. In this model, we set the
    grid space domain to :math:`G = [1 : N_1] \\times [1 : 2 N_2 + 1]`, where
    the odd number of points in the :math:`s_2`-direction is to include the
    stagnation streamline. Let :math:`\\Delta s_1 = L_1 / (N_1 - 1)` and
    :math:`\\Delta s_2 = 2 L_2 / (2 N_2)`. Here, we let the state be :math:`x =
    (\\psi_{i, j}, \\omega_{i, j})_{(i, j) \\in G}`. Of course, one could cut
    out the boundary points since they are determined by the fluid points.
    Here, for computational simplicity, we will keep the boundary points in the
    state. Let :math:`x` be a state satisfying the boundary conditions. Please
    see ``gen_ic()`` for an example on how to generate such a state. Given a
    state :math:`x`, we can generate the velocity field using second-order
    finite-difference approximations of the derivatives. In particular, for
    :math:`(i, j) \\in G - \\partial G`, we can calculate the velocity field
    using

    .. math::
        \\begin{align*}
            u_{1, i, j} = \\frac{\\psi_{i, j + 1} - \\psi_{i, j - 1}}{2
            \\Delta s_2}, \\quad
            u_{2, i, j} = - \\frac{\\psi_{i + 1, j} - \\psi_{i - 1, j}}{2
            \\Delta s_1}.
        \\end{align*}

    The velocity field on the top, left, and bottom boundary points is set by
    using :math:`\\text{BC1}` through :math:`\\text{BC3}`. The cylinder
    boundary points are defined by :math:`\\text{BC5}`. The right boundary
    points of :math:`u_{2, i, j}` are calculated using the left-weighted
    second-order approximation

    .. math::
        \\begin{align*}
            u_{2, i, j} = - \\frac{3 \\psi_{i, j} - 4 \\psi_{i - 1, j}
            + \\psi_{i - 2, j}}{2 \\Delta s_1}.
        \\end{align*}

    Next, lets calculate the updated flow points of vorticity. We can now
    calculate the update of the vorticity field :math:`\\omega_{i, j}` on
    :math:`G - \\partial G`. Here, we use Euler's method to update in time, we
    use upwind differencing to calculate the advection term, and we use
    central differencing to calculate the diffusion term. The update is
    given explicitly by

    .. math::
        \\begin{align*}
            \\omega_{i, j} + \\Delta t \\left(
                - \\text{UppDiff}_{s_1} (u_1, \\omega)
                - \\text{UppDiff}_{s_2} (u_2, \\omega)
                + \\nu \\text{Lap} (\\omega)
            \\right)
        \\end{align*}

    where

    .. math::
        \\begin{align*}
            \\text{UppDiff}_{s_1} (u_1, \\omega) &=
            \\begin{cases}
                \\frac{u_{i, j} \\omega_{i, j} - u_{1, i - 1, j}
                \\omega_{i - 1, j}}{\\Delta s_1} & \\text{if } u_{1, i, j} > 0
                \\\\
                \\frac{u_{1, i + 1, j} \\omega_{i + 1, j}
                - u_{1, i, j} \\omega_{i, j}}{\\Delta s_1} & \\text{if }
                u_{1, i, j} < 0
            \\end{cases} \\\\
            \\text{UppDiff}_{s_2} (u_2, \\omega) &=
            \\begin{cases}
                \\frac{u_{2, i, j} \\omega_{i, j} - v_{i, j - 1}
                \\omega_{i, j - 1}}{\\Delta s_2} & \\text{if }
                u_{2, i, j} > 0 \\\\
                \\frac{u_{2, i, j + 1} \\omega_{i, j + 1}
                - u_{2, i, j} \\omega_{i, j}}{\\Delta s_2} & \\text{if }
                u_{2, i, j} < 0
            \\end{cases} \\\\
            \\text{Lap} (\\omega) &= \\frac{\\omega_{i + 1, j}
            - 2 \\omega_{i, j} + \\omega_{i - 1, j}}{\\Delta s_1^2}
            + \\frac{\\omega_{i, j + 1} - 2 \\omega_{i, j}
            + \\omega_{i, j - 1}}{\\Delta s_2^2}.
        \\end{align*}

    Given a state :math:`x` we generate the vorticity :math:`\\omega_{i, j}`
    on the boundaries by (1) setting the left, top, and bottom boundary
    conditions to zero and (2) determining the right boundary by using
    :math:`\\text{BC4a}`:

    .. math::
        \\begin{align*}
            U_\\infty \\partial \\omega / \\partial s_1 \\approx
            \\nu (\\nabla_s \\cdot \\nabla_s) \\omega
        \\end{align*}

    .. math::
        \\begin{align*}
            U_\\infty \\frac{
                3 \\omega_{i,j} - 4 \\omega_{i - 1,j} + \\omega_{i - 2,j}
            }{2 \\Delta s_1}
            &\\approx \\nu
            \\frac{2 \\omega_{i, j} - 5 \\omega_{i - 1, j}
            + 4 \\omega_{i - 2, j} - \\omega_{i - 3, j}}{\\Delta s_1^3} \\\\
            & \\qquad + \\: \\nu \\frac{\\omega_{i, j + 1} - 2 \\omega_{i, j}
            + \\omega_{i, j - 1}}{\\Delta s_2^2}
        \\end{align*}

    .. math::
        \\begin{align*}
            \\omega_{i,j}
            &\\approx \\frac{U_\\infty \\left(\\frac{4 \\omega_{i-1, j}
            - \\omega_{i-2, j}}{2 \\Delta s_1}\\right)
            + \\nu \\left( \\frac{-5 \\omega_{i-1, j} + 4 \\omega_{i-2, j}
            - \\omega_{i-3,j}}{\\Delta s_1^3} + \\frac{\\omega_{i, j+1}
            + \\omega_{i, j-1}}{\\Delta s_2^2}\\right)}{
            \\frac{3 U_\\infty}{2 \\Delta s_1} - \\frac{2 \\nu}{\\Delta s_1^3}
            + \\frac{2 \\nu}{\\Delta s_2^2}
            },
        \\end{align*}

    where :math:`i = N_1` and :math:`j \\in [2 : 2 N_2]`. We now have the
    tensor :math:`\\omega_{i, j}` everywhere but the cylinder surface. From
    this, we solve the non-boundary points of :math:`\\psi_{i, j}` by solving
    the Poisson equation. This was done using the Jacobi method defined by

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
    because we found that the tensorized Jacobi method was faster than
    Gauss-Seidel. We then directly enforced the boundary point values defined
    in :math:`\\text{BC1}` through :math:`\\text{BC3}`. The right boundary
    points are calculated by using the left-weighted second-order approximation
    of :math:`\\partial \\psi / \\partial s_1 \\approx 0`:

    .. math::
        \\begin{align*}
            \\frac{3 \\psi_{i, j} - 4 \\psi_{i - 1, j} + \\psi_{i - 2, j}}
            {2 \\Delta s_1} \\approx 0.
        \\end{align*}

    Finally, we calculate the values of :math:`\\omega` on the boundary of the
    cylinder. Let :math:`(i, j)` be a corner point on the cylinder surface
    with :math:`(i + 1, j)` and :math:`(i, j + 1)` be two fluid points. Using
    the fact that :math:`\\nabla \\psi = 0` on the  cylinder surface, we
    make following second-order approximations:

    .. math::
        \\begin{align*}
            \\psi_{i + 1, j} &= \\psi_{i, j} + 0
            + \\frac{\\Delta s_1^2}{2}
            \\frac{\\partial^2 \\psi}{\\partial s_1^2} \\\\
            \\psi_{i, j + 1} &= \\psi_{i, j} + 0
            + \\frac{\\Delta s_2^2}{2}
            \\frac{\\partial^2 \\psi}{\\partial s_2^2}.
        \\end{align*}

    .. math::
        \\begin{align*}
            \\omega_{i, j} = - 2 \\left(\\frac{\\psi_{i + 1, j}}{\\Delta s_1^2}
            + \\frac{\\psi_{i, j + 1}}{\\Delta s_2^2}.
            \\right)
        \\end{align*}

    If we let :math:`\\psi_{i, j} = 0` inside the fluid surface, then one will
    see, after looking at each possible boundary point combination of the
    cylinder, that the values of :math:`\\omega_{i, j}` are given by

    .. math::
        \\begin{align*}
            \\omega_{i, j} = - 2 \\left(\\frac{\\psi_{i + 1, j}
            + \\psi_{i - 1, j}}{\\Delta s_1^2}
            + \\frac{\\psi_{i, j + 1} + \\psi_{1, j - 1}}{\\Delta s_2^2}.
            \\right)
        \\end{align*}

    This determines all values of the next state :math:`\\Phi_{\\Delta t}(x)`.

    Finally, this class enforces two conditions: First, the grid spacings
    must satisfy a condition for good diffusion modeling:

    .. math::
        \\begin{align*}
            Re_{\\Delta s_1} = \\frac{U_\\infty \\Delta s_1}{\\nu} < 10,
            \\quad
            Re_{\\Delta s_2} = \\frac{U_\\infty \\Delta s_2}{\\nu} < 10.
        \\end{align*}

    Second, the time step must satisfy the Courant-Friedrichs-Lewy condition:

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
    |   ``L_1`` (``float``): the value of the parameter :math:`L_1`
    |   ``L_2`` (``float``): the value of the parameter :math:`L_2`
    |   ``N_1`` (``int``): the value of the parameter :math:`N_1`
    |   ``N_2`` (``int``): the value of the parameter :math:`N_2`
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
    |   [1] 1986 - Girault - Finite Element Methods for Navier-Stokes
            Equations: Theory and Algorithms
    |   [2] 1994 - Quarteroni - Numerical Approximation of Partial Differential
            Equations
    |   [3] 1998 - Störtkuhl - On Boundary Conditions for the Vorticity-Stream
            Function Formulation
    |   [4] 1999 - Turek - Efficient Solvers for Incompressible Flow Problems
            An Algorithmic and Computational Approach
    |   [5] 2000 - Gresho - Incompressible Flow and the Finite Element Method
    |   [6] 2011 - Yew - Numerical Differentiation Finite Differences
    |   [7] 2016 - John - Finite Element Methods for Incompressible Flow
            Problems
    |   [8] 2025 - Nosenchuck - Lectures in MAE 423 Heat Transfer
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
        # pull out the stream function and vorticity
        psi_prev = x[..., 0]
        omega_prev = x[..., 1]
        # update the fluid points of vorticity
        u1, u2 = self._vel(psi_prev)
        omega = zeros(x.shape[:-len(self.dims_state)] + (self.m, self.n))
        w_i_j = omega_prev[..., 1:self.m-1, 1:self.n-1]
        w_ip1_j = omega_prev[..., 1:self.m-1, 2:self.n]
        w_im1_j = omega_prev[..., 1:self.m-1, 0:self.n-2]
        w_i_jp1 = omega_prev[..., 0:self.m-2, 1:self.n-1]
        w_i_jm1 = omega_prev[..., 2:self.m, 1:self.n-1]
        lap = ((w_ip1_j - 2 * w_i_j + w_im1_j) / self.ds1**2
               + (w_i_jp1 - 2 * w_i_j + w_i_jm1) / self.ds2**2)
        u1_i_j = u1[..., 1:self.m-1, 1:self.n-1]
        u1_ip1_j = u1[..., 1:self.m-1, 2:self.n]
        u1_im1_j = u1[..., 1:self.m-1, 0:self.n-2]
        Dwu1_neg = (u1_ip1_j * w_ip1_j - u1_i_j * w_i_j)
        Dwu1_pos = (u1_i_j * w_i_j - u1_im1_j * w_im1_j)
        Dwu1 = where(u1[1:self.m-1, 1:self.n-1] > 0.0, Dwu1_pos, Dwu1_neg)
        u2_i_j = u2[..., 1:self.m-1, 1:self.n-1]
        u2_i_jp1 = u2[..., 0:self.m-2, 1:self.n-1]
        u2_i_jm1 = u2[..., 2:self.m, 1:self.n-1]
        Dwu2_neg = (u2_i_jp1 * w_i_jp1 - u2_i_j * w_i_j)
        Dwu2_pos = (u2_i_j * w_i_j - u2_i_jm1 * w_i_jm1)
        Dwu2 = where(u2[1:self.m-1, 1:self.n-1] > 0.0, Dwu2_pos, Dwu2_neg)
        omega[..., 1:self.m-1, 1:self.n-1] = (w_i_j
                                              + self.dt * (- Dwu1 / self.ds1
                                                           - Dwu2 / self.ds2
                                                           + (self.nu * lap)))
        omega = where(self.mask == 0, omega, 0.0)
        # update the fluid points of stream function
        psi = self._poisson(psi_prev, omega)
        # update the boundary points of stream function
        psi = self._psi_bd(psi)
        # update the boundary points of vorticity
        omega = self._omega_bd(psi, omega)
        return stack((psi, omega), dim=-1)

    def __init__(self, L_1: float = 2.0, L_2: float = 0.25,  # noqa: C901
                 N_1: int = 1200, N_2: int = 150, nu: float = 1.57e-5,
                 R: float = 0.05, Re: float = 200.0,
                 h_rf: float = 2.0/5.0, dt_rf: float = 0.5,
                 ep: float = 1e-7, s_c1: float = 0.2) -> None:
        """Initialize the superclass and model parameters.

        This method initializes the superclass and model parameters.

        | **Args**
        |   ``L_1`` (``float``): the value of the parameter :math:`L_1` with
                default value of ``2.0``
        |   ``L_2`` (``float``): the value of the parameter :math:`L_2` with
                default value of ``0.25``
        |   ``N_1`` (``float``): the value of the parameter :math:`N_1` with
                default value of ``1200``
        |   ``N_2`` (``float``): the value of the parameter :math:`N_2`
                with default value of ``150``
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
        |   ``s_c1`` (``float``): the value of the parameter :math:`s_{c1}`
                with default value of ``0.2``

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
        self.L_1 = L_1
        self.L_2 = L_2
        self.N_1 = N_1
        self.N_2 = N_2
        self.m = 2 * self.N_2 + 1
        self.n = self.N_1
        self._dims_state = (self.m, self.n, 2)
        self.nu = nu
        # set the continuous space parameters
        self.R = R
        D = 2 * self.R
        self.U_inf = Re * nu / D
        self.ds2 = 2 * L_2 / (2 * N_2)
        self.ds1 = L_1 / (N_1 - 1)
        h = 10 * self.nu / self.U_inf * h_rf
        if (self.ds2 > h) or (self.ds1 > h):
            raise ValueError('Grid space must be smaller than 10 * self.nu / '
                             + f'self.U_inf * h_rf = {h:.4f}. Currently, '
                             + f'self.ds2 = {self.ds2:.4f} and '
                             + f'self.ds1 = {self.ds1:.4f}.')
        self.dt = (min(self.ds2, self.ds1) / (2 * self.U_inf)) * dt_rf
        self.ep = ep
        self.s_c = (s_c1, L_2)
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
        # store list of grid points corresponding to the cylinder boundary
        self.bd = []
        for i in range(self.m):
            for j in range(self.n):
                if self.mask[i, j] == 2.0:
                    self.bd.append((i, j))
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
        # define the initial guess for the stream function
        psi = zeros((self.m, self.n))
        for i in range(self.m):
            for j in range(self.n):
                if sqrt((i * self.ds2 - self.s_c[1])**2
                        + (j * self.ds1 - self.s_c[0])**2) > self.R:
                    psi[i, j] = (- self.U_inf * (i * self.ds2 - self.s_c[1]))
                else:
                    psi[i, j] = 0.0
        # initialize a random vorticity field
        omega = zeros((self.m, self.n))  # 2 * rand((self.m, self.n)) - 1
        # use the omega fluid points to set psi fluid points
        psi = self._poisson(psi, omega)
        # enforce psi right boundary condition
        psi = self._psi_bd(psi)
        # enforce the cylinder boundary condition for vorticity
        omega = self._omega_bd(psi, omega)
        return stack((psi, omega), dim=-1)

    def _poisson(self, psi: Tensor, omega: Tensor) -> Tensor:
        # set up the initial guess for the stream function
        psi_prev = psi.clone()
        psi_curr = psi.clone()
        error = inf
        # do Jacobi iteration
        mult = 1 / (2 * (1 / self.ds1**2 + 1 / self.ds2**2))
        while error > self.ep:
            term1 = (psi_prev[..., 1:self.m-1, 2:self.n]
                     + psi_prev[..., 1:self.m-1, 0:self.n-2]) / self.ds1**2
            term2 = (psi_prev[..., 2:self.m, 1:self.n-1]
                     + psi_prev[..., 0:self.m-2, 1:self.n-1]) / self.ds2**2
            term3 = omega[..., 1:self.m-1, 1:self.n-1]
            psi_curr[..., 1:self.m-1, 1:self.n-1] = mult * (term1
                                                            + term2
                                                            + term3)
            psi_curr = where(self.mask == 0, psi_curr, 0.0)
            with no_grad():
                error = max(abs(psi_curr - psi_prev)).item()
            psi_prev = psi_curr.clone()
        # return the stream function
        return psi_curr

    def _psi_bd(self, psi: Tensor) -> Tensor:
        # set the right boundary condition
        psi_new = psi.clone()
        psi_new[..., 1:self.m-1, -1] = (1 / 3) * (4 * psi[..., 1:self.m-1, -2]
                                                  - psi[..., 1:self.m-1, -3])
        # return the stream function
        return psi_new

    def _omega_bd(self, psi: Tensor, omega: Tensor) -> Tensor:
        # set the right boundary condition
        omega_new = omega.clone()
        im1_j = omega[..., 1:self.m-1, -2]
        im2_j = omega[..., 1:self.m-1, -3]
        im3_j = omega[..., 1:self.m-1, -4]
        i_jp1 = omega[..., 0:self.m-2, -1]
        i_jm1 = omega[..., 2:self.m, -1]
        term1 = self.U_inf * ((4 * im1_j - im2_j) / (2 * self.ds1))
        term2 = (self.nu * ((-5 * im1_j + 4 * im2_j - im3_j) / self.ds1**3
                            + (i_jp1 + i_jm1) / self.ds2**2))
        term3 = (3 * self.U_inf / (2 * self.ds1)
                 - 2 * self.nu / self.ds1**3 + 2 * self.nu / self.ds2**2)
        omega_new[..., 1:self.m-1, -1] = (term1 + term2) / term3
        # set the cylinder boundary condition
        for ij in self.bd:
            term1 = (psi[..., ij[0] + 1, ij[1]] - 2 * psi[..., ij[0], ij[1]]
                     + psi[..., ij[0] - 1, ij[1]]) / self.ds1**2
            term2 = (psi[..., ij[0], ij[1] + 1] - 2 * psi[..., ij[0], ij[1]]
                     + psi[..., ij[0], ij[1] - 1]) / self.ds2**2
            omega_new[..., ij[0], ij[1]] = - 2 * (term1 + term2)
        return omega_new

    def _vel(self, psi: Tensor) -> Tuple[Tensor, Tensor]:
        # compute the u1 velocity field
        u1 = zeros((self.m, self.n))
        u1[0, :] = self.U_inf
        u1[self.m - 1, :] = self.U_inf
        u1[:, 0] = self.U_inf
        u1[1:self.m-1, 1:] = (psi[..., 0:self.m-2, 1:]
                              - psi[..., 2:self.m, 1:]) / (2 * self.ds2)
        u1 = where(self.mask == 0, u1, 0.0)
        # compute the u_2 velocity field
        u2 = zeros((self.m, self.n))
        u2[0, :] = 0.0
        u2[self.m - 1, :] = 0.0
        u2[:, 0] = 0.0
        u2[1:-1, 1:self.n-1] = (psi[..., 1:-1, 2:self.n]
                                - psi[..., 1:-1, 0:self.n-2]) / (2 * self.ds1)
        u2[1:-1, self.n-1] = (3 * psi[..., 1:-1, self.n-1]
                              - 4 * psi[..., 1:-1, self.n-2]
                              + psi[..., 1:-1, self.n-3]) / (2 * self.ds1)
        u2 = -1 * where(self.mask == 0, u2, 0.0)
        # return the (u1, u2)
        return u1, u2
