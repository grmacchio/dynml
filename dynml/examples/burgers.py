"""Contain all code related to the Burgers' dynamical system.

This module contains all code related to the Burgers' dynamical system.
"""


# import built-in python-package code
# None
# import external python-package code
# None
# import internal python-package code
from dynml.dyn.cont.ode.firstorder import SemiLinearFirstOrderSystem


# export public code
__all__ = ['Burgers']


# define the dynamical system
class Burgers(SemiLinearFirstOrderSystem):
    """Represent the discretized Burgers' dynamical system.

    This class represents the discretized Burgers' dynamical system. In
    particular, we examine the discretized Burgers' dynamical system with
    periodic boundary conditions on :math:`[0, 2\\pi]`. Burgers' equation
    is written as

    .. math::
        \\frac{\\partial u}{\\partial t} =
        \\nu \\frac{\\partial^2 u}{\\partial x^2}
        - u \\frac{\\partial u}{\\partial x},

    where :math:`u(t, x) \\in \\mathbb{R}` is the velocity field, :math:`t \\in
    \\mathbb{R}` is the time, :math:`x \\in [0, 2\\pi]` is the spatial
    coordinate, and :math:`\\nu > 0` is the viscosity.
    """
