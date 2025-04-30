"""Define and utilize various time-stepping methods for time discretization.

This module defines and utilizes various time-stepping methods for time
discretization. A time-stepping method is a finite-difference method for
first-order ordinary-differential-equation time discretization.
"""


# import built-in python-package code
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Type, Union
# import external python-package code
from torch import diag, eye, Tensor
from torch.linalg import inv, lu_factor, lu_solve, norm
from torch.nn import Module, ModuleList, Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.system import FirstOrderSystem
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem
from dynml.dyn.discrete.system import DiscreteSystem


# export public code
__all__ = ["ExplicitTimeStepMethod",
           "Euler", "RK2", "RK4",

           "LinearSolver",
           "LU", "Inv", "Diag",

           "SemiLinearTimeStepMethod",
           "RK2CN", "RK3CN",

           "DiscretizedFirstOrderSystem",
           "gen_approx_discrete_sys"]


# define various time-stepping methods for time discretization
class ExplicitTimeStepMethod(ABC, Module):
    """Represent an explicit time-stepping method.

    This abstract registry class represents an explicit time-stepping method.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["ExplicitTimeStepMethod"]]``): a
            registry of explicit time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

    | **Abstract Methods**
    |   ``approx_flow_map()``: the ``self.dt`` approximate flow map

    | **Class Methods**
    |   ``methods()``: return the list of available explicit time-stepping
            methods
    |   ``lookup()``: return the subclass whose name matches the input string
            key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``__init__()``: initialize the superclasses and attributes

    | **References**
    |   None
    """

    registry: Dict[str, Type["ExplicitTimeStepMethod"]] = {}

    @abstractmethod
    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The ``self.dt`` approximate flow map.

        This method represents the ``self.dt`` approximate flow map.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (self.sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    @classmethod
    def methods(cls) -> List[str]:
        """Return the list of available explicit time-stepping methods.

        This method returns the list of available explicit time-stepping
        methods.

        | **Args**
        |   None

        | **Return**
        |   ``List[str]``: the list of available explicit time-stepping methods

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the list of available explicit time-stepping methods
        return list(cls.registry.keys())

    @classmethod
    def lookup(cls, method: str) -> Type["ExplicitTimeStepMethod"]:
        """Return the subclass whose name matches the input string key.

        This method returns the subclass whose name matches the input string
        key. In particular, this method retrieves a specific explicit
        time-stepping method using the input string key and the
        ``cls.registry`` attribute .

        | **Args**
        |   ``method`` (``str``): the string key

        | **Return**
        |   ``Type["ExplicitTimeStepMethod"]``: the subclass corresponding to
                the given string key

        | **Raises**
        |   ``NotImplementedError``: if the method is unknown

        | **References**
        |   None
        """
        # return the subclass corresponding to the given string key
        try:
            return cls.registry[method.lower()]
        # raise an error if the method is unknown
        except KeyError as exc:
            raise NotImplementedError(f"'{method}' is an unknown "
                                      + "time-stepping method") from exc

    def __init_subclass__(cls) -> None:
        """Register subclasses by their lowercase name.

        This method registers subclasses by their lowercase name.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # register subclasses by their lowercase name
        name = cls.__name__.lower()
        cls.registry[name] = cls

    def __init__(self, dt: float, sys: FirstOrderSystem) -> None:
        """Initialize the superclasses and attributes.

        This method initializes the ``ABC`` and ``Module`` superclasses and the
        attributes.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

        | **Return**
        |  None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclasses and attributes
        super().__init__()
        self.dt = dt
        self.sys = sys
        # set computationally efficient right hand side
        if isinstance(sys, SemiLinearFirstOrderSystem):
            if norm(self.sys.A - diag(diag(self.sys.A)), ord='fro') == 0.0:
                self._diag_A = diag(sys.A)
                self._rhs = lambda x: x * self._diag_A + self.sys.nonlinear(x)
            else:
                self._rhs = sys.rhs
        else:
            self._rhs = sys.rhs


class Euler(ExplicitTimeStepMethod):
    """Represent the Euler explicit time-stepping method.

    This class represents the Euler explicit time-stepping method. See [1] for
    more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["ExplicitTimeStepMethod"]]``): a
            registry of explicit time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available explicit time-stepping
            methods
    |   ``lookup()``: return the subclass whose name matches the input string
            key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``approx_flow_map()``: the Euler's method ``self.dt`` approximate flow
            map
    |   ``__init__()``: initialize the superclass

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 143.
    """

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The Euler's method ``self.dt`` approximate flow map.

        This method represents the Euler's method ``self.dt`` approximate flow
        map.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 143.
        """
        # use Euler's method to approx. advance the state by one time-step
        return x + self.dt * self._rhs(x)

    def __init__(self, dt: float, sys: FirstOrderSystem) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``ExplicitTimeStepMethod``.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass
        super().__init__(dt, sys)


class RK2(ExplicitTimeStepMethod):
    """Define the second-order Runge-Kutta explicit time-stepping method.

    This class defines the second-order Runge-Kutta (R.K.2.) explicit
    time-stepping method. See [1] for more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["ExplicitTimeStepMethod"]]``): a
            registry of explicit time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available explicit time-stepping
            methods
    |   ``lookup()``: return the subclass whose name matches the input string
            key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``approx_flow_map()``: the R.K.2. method ``self.dt`` approximate flow
            map
    |   ``__init__()``: initialize the superclass

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 143.
    """

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The R.K.2. method ``self.dt`` approximate flow map.

        This method represents the R.K.2. method ``self.dt`` approximate flow
        map.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 143.
        """
        # use R.K.2. method to approx. advance the state by one time-step
        k1 = self.dt * self._rhs(x)
        k2 = self.dt * self._rhs(x + k1)
        return x + (k1 + k2) / 2.0

    def __init__(self, dt: float, sys: FirstOrderSystem) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``ExplicitTimeStepMethod``.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass
        super().__init__(dt, sys)


class RK4(ExplicitTimeStepMethod):
    """Define the forth-order Runge-Kutta explicit time-stepping method.

    This class defines the forth-order Runge-Kutta (R.K.4.) explicit
    time-stepping method. See [1] for more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["ExplicitTimeStepMethod"]]``): a
            registry of explicit time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available explicit time-stepping
            methods
    |   ``lookup()``: return the subclass whose name matches the input string
            key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``approx_flow_map()``: the R.K.4. method ``self.dt`` approximate flow
            map
    |   ``__init__()``: initialize the superclass

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 144.
    """

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The R.K.4. method ``self.dt`` approximate flow map.

        This method represents the R.K.4. method ``self.dt`` approximate flow
        map.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 144.
        """
        # use R.K.4. method to approx. advance the state by one time-step
        k1 = self.dt * self._rhs(x)
        k2 = self.dt * self._rhs(x + k1 / 2.0)
        k3 = self.dt * self._rhs(x + k2 / 2.0)
        k4 = self.dt * self._rhs(x + k3)
        return x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

    def __init__(self, dt: float, sys: FirstOrderSystem) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``ExplicitTimeStepMethod``.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # call the superclass
        super().__init__(dt, sys)


class LinearSolver(ABC, Module):
    """Represent a linear solver.

    This abstract registry class represents a linear solver. A linear solver
    is a method for solving a linear system of equations: :math:`x^TM^T = y^T`.
    The shape of the matrix ``M`` is ``(n, n)``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["LinearSolver"]]``): a registry of
            linear solvers

    | **Attributes**
    |   None

    | **Abstract Methods**
    |   ``__init__()``: compute matrix factorizations and initialize the
            superclasses
    |   ``__call__()``: return the solution :math:`x^T` given the input
            :math:`y^T`

    | **Class Methods**
    |   ``methods()``: return the list of available linear solvers
    |   ``lookup()``: return the subclass whose name matches the input string
            key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   None

    | **References**
    |   None
    """

    registry: Dict[str, Type["LinearSolver"]] = {}

    @abstractmethod
    def __init__(self, M: Tensor):
        """Compute matrix factorizations and initialize the superclasses.

        This method computes matrix factorizations and initializes the
        superclasses. Remember to store the factorizations as
        ``torch.nn.Parameter`` attributes.

        | **Args**
        |   ``M`` (``Tensor``): the matrix ``M`` in the linear system with
                shape ``(n, n)``

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclasses to enable torch.nn.Module functionality
        super().__init__()
        # code pass as a placeholder for future implementation
        pass

    @abstractmethod
    def __call__(self, yT: Tensor) -> Tensor:
        """Return the solution :math:`x^T` given the input :math:`y^T`.

        This method returns the solution :math:`x` given the input :math:`y`.
        Here, the input and solution satisfy the linear system of equations:
        :math:`x^TM^T = y^T`.

        | **Args**
        |   ``yT`` (``Tensor``): the input :math:`y^T` with shape
                ``(...,) + (n,)``

        | **Return**
        |   ``Tensor``: the solution :math:`x^T` with shape
                ``(...,) + (n,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    @classmethod
    def methods(cls) -> List[str]:
        """Return the list of available linear solvers.

        This method returns the list of available linear solvers.

        | **Args**
        |   None

        | **Return**
        |   ``List[str]``: the list of available linear solvers

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the list of available linear solvers
        return list(cls.registry.keys())

    @classmethod
    def lookup(cls, method: str) -> Type["LinearSolver"]:
        """Return the subclass whose name matches the input string key.

        This method returns the subclass whose name matches the input string
        key. In particular, this method retrieves a specific linear solver
        using the input string key and the ``cls.registry`` attribute.

        | **Args**
        |   ``method`` (``str``): the string key

        | **Return**
        |   ``Type["LinearSolver"]``: the subclass corresponding to the given
                string key

        | **Raises**
        |   ``NotImplementedError``: if the method is unknown

        | **References**
        |   None
        """
        # return the subclass corresponding to the given string key
        try:
            return cls.registry[method.lower()]
        # raise an error if the method is unknown
        except KeyError as exc:
            raise NotImplementedError(f"'{method}' is an unknown "
                                      + "linear solver") from exc

    def __init_subclass__(cls) -> None:
        """Register subclasses by their lowercase name.

        This method registers subclasses by their lowercase name.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # register subclasses by their lowercase name
        name = cls.__name__.lower()
        cls.registry[name] = cls


class LU(LinearSolver):
    """Represent the lower-upper (L.U.) decomposition linear solver.

    This class represents the lower-upper (L.U.) decomposition linear solver.
    Here, we are interested in solving the linear system of equations:
    :math:`x^TM^T = y^T` where the matrix ``M`` is ``(n, n)``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["LinearSolver"]]``): a registry
            of linear solvers

    | **Attributes**
    |   ``LU`` (``Parameter``): L.U. decomposition of the matrix ``M``
    |   ``pivots`` (``Parameter``): the pivots of the L.U. decomposition of the
            matrix ``M``

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available linear solvers
    |   ``lookup()``: return the subclass whose name matches the input
            string key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``__init__()``: compute the L.U. decomposition and initialize the
            superclasses
    |   ``__call__()``: return the solution :math:`x^T` given the input
            :math:`y^T`

    | **References**
    |   None
    """

    def __init__(self, M: Tensor):
        """Compute the L.U. decomposition and initialize the superclasses.

        This method computes the L.U. decomposition and initializes the
        superclasses.

        | **Args**
        |   ``M`` (``Tensor``): the matrix ``M`` in the linear system with
                shape ``(n, n)``

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # compute the L.U. decomposition and initialize the superclasses
        super().__init__(M)
        LU, pivots = lu_factor(M)
        self.LU = Parameter(LU, requires_grad=False)
        self.pivots = Parameter(pivots, requires_grad=False)

    def __call__(self, yT: Tensor) -> Tensor:
        """Return the solution :math:`x^T` given the input :math:`y^T`.

        This method returns the solution :math:`x^T` given the input
        :math:`y^T`. Here, the input and solution satisfy the linear system of
        equations: :math:`x^TM^T = y^T`.

        | **Args**
        |   ``y`` (``Tensor``): the input :math:`y^T` with shape
                ``(...,) + (n,)``

        | **Return**
        |   ``Tensor``: the solution :math:`x^T` with shape
                ``(...,) + (n,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the solution x given the input y
        return lu_solve(self.LU, self.pivots,
                        yT.transpose(-1, -2)).transpose(-1, -2)


class Inv(LinearSolver):
    """Represent the inverse linear solver.

    This class represents the inverse linear solver. Here, we are interested in
    solving the linear system of equations: :math:`x^TM^T = y^T` where the
    matrix ``M`` is ``(n, n)``.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["LinearSolver"]]``): a registry
            of linear solvers

    | **Attributes**
    |   ``M_inv`` (``Parameter``): the inverse of the matrix ``M``

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available linear solvers
    |   ``lookup()``: return the subclass whose name matches the input
            string key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``__init__()``: compute the inverse and initialize the superclasses
    |   ``__call__()``: return the solution :math:`x^T` given the input
            :math:`y^T`

    | **References**
    |   None
    """

    def __init__(self, M: Tensor):
        """Compute the inverse and initialize the superclasses.

        This method computes the inverse and initializes the superclasses.

        | **Args**
        |   ``M`` (``Tensor``): the matrix ``M`` in the linear system with
                shape ``(n, n)``

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # compute the inverse and initialize the superclasses
        super().__init__(M)
        self.M_inv = Parameter(inv(M), requires_grad=False)

    def __call__(self, yT: Tensor) -> Tensor:
        """Return the solution :math:`x^T` given the input :math:`y^T`.

        This method returns the solution :math:`x^T` given the input
        :math:`y^T`. Here, the input and solution satisfy the linear system of
        equations: :math:`x^TM^T = y^T`.

        | **Args**
        |   ``y`` (``Tensor``): the input :math:`y^T` with shape
                ``(...,) + (n,)``

        | **Return**
        |   ``Tensor``: the solution :math:`x^T` with shape
                ``(...,) + (n,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the solution x given the input y
        return yT @ self.M_inv


class Diag(LinearSolver):
    """Represent the diagonal-matrix linear solver.

    This class represents the diagonal-matrix linear solver. Here, we are
    interested in solving the linear system of equations: :math:`x^TM^T = y^T`
    where the matrix ``M`` is ``(n, n)`` and diagonal.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["LinearSolver"]]``): a registry
            of linear solvers

    | **Attributes**
    |   ``d_inv`` (``Parameter``): the diagonal of the inverse of ``M``

    | **Abstract Methods**
        |   None

    | **Class Methods**
    |   ``methods()``: return the list of available linear solvers
    |   ``lookup()``: return the subclass whose name matches the input
            string key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``__init__()``: compute the diagonal inverse and initialize the
            superclasses
    |   ``__call__()``: return the solution :math:`x^T` given the input
            :math:`y^T`

    | **References**
    |   None
    """

    def __init__(self, M: Tensor):
        """Compute the diagonal of the inverse and initialize the superclasses.

        This method computes the diagonal of the inverse and initializes the
        superclasses.

        | **Args**
        |   ``M`` (``Tensor``): the matrix ``M`` in the linear system with
                shape ``(n, n)``

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # compute the diagonal of the inverse and initialize the superclasses
        super().__init__(M)
        self.d_inv = Parameter(1.0 / diag(M), requires_grad=False)

    def __call__(self, yT: Tensor) -> Tensor:
        """Return the solution :math:`x^T` given the input :math:`y^T`.

        This method returns the solution :math:`x^T` given the input
        :math:`y^T`. Here, the input and solution satisfy the linear system of
        equations: :math:`x^TM^T = y^T`.

        | **Args**
        |   ``y`` (``Tensor``): the input :math:`y^T` with shape
                ``(...,) + (n,)``

        | **Return**
        |   ``Tensor``: the solution :math:`x^T` with shape
                ``(...,) + (n,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the solution x given the input y
        return yT * self.d_inv


class SemiLinearTimeStepMethod(ABC, Module):
    """Represent a semi-linear time-stepping method.

    This abstract registry class represents a semi-linear time-stepping method.
    A semi-linear time-stepping method is a semi-implicit time-stepping method
    applied to a semi-linear first-order ordinary-differential-equation system,
    where the linear part is time discretized implicitly and the nonlinear part
    is time discretized explicitly. See [1] for more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["SemiLinearTimeStepMethod"]]``):
            a registry of semi-linear time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear first-order
            O.D.E. system

    | **Abstract Methods**
    |   ``init_implicit_solvers()``: initialize the implicit solvers
    |   ``approx_flow_map()``: the ``self.dt`` approximate flow map

    | **Class Methods**
    |   ``methods()``: return the list of available semi-linear time-stepping
            methods
    |   ``lookup()``: return the subclass whose name matches the input string
            key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``__init__()``: initialize the superclasses and attributes

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 148-149.
    """

    registry: Dict[str, Type["SemiLinearTimeStepMethod"]] = {}

    @abstractmethod
    def init_implicit_solvers(self, implicit_solver_cls: Type["LinearSolver"]
                              ) -> None:
        """Initialize the implicit solvers.

        This method initializes the implicit solvers. In particular, this
        method initializes the attribute ``self.implicit_solver_insts``, which
        is a ``ModuleList`` of ``LinearSolver`` s used in
        ``self.approx_flow_map()``.

        | **Args**
        |   ``implicit_solver_cls`` (``Type["LinearSolver"]``): the implicit
                solver class

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    @abstractmethod
    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The ``self.dt`` approximate flow map.

        This method represents the ``self.dt`` approximate flow map.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # code pass as a placeholder for future implementation
        pass

    @classmethod
    def methods(cls) -> List[str]:
        """Return the list of available semi-linear time-stepping methods.

        This method returns the list of available semi-linear time-stepping
        methods.

        | **Args**
        |   None

        | **Return**
        |   ``List[str]``: the list of available semi-linear time-stepping
                methods

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # return the list of available explicit time-stepping methods
        return list(cls.registry.keys())

    @classmethod
    def lookup(cls, method: str) -> Type["SemiLinearTimeStepMethod"]:
        """Return the subclass whose name matches the input string key.

        This method returns the subclass whose name matches the input string
        key. In particular, this method retrieves a specific semi-linear
        time-stepping method using the input string key and the
        ``cls.registry`` attribute.

        | **Args**
        |   ``method`` (``str``): the string key

        | **Return**
        |   ``Type["SemiLinearTimeStepMethod"]``: the subclass corresponding to
                the given string key

        | **Raises**
        |   ``NotImplementedError``: if the method is unknown

        | **References**
        |   None
        """
        # return the subclass corresponding to the given string key
        try:
            return cls.registry[method.lower()]
        # raise an error if the method is unknown
        except KeyError as exc:
            raise NotImplementedError(f"'{method}' is an unknown "
                                      + "semi-linear time-stepping "
                                      + "method") from exc

    def __init_subclass__(cls) -> None:
        """Register subclasses by their lowercase name.

        This method registers subclasses by their lowercase name.

        | **Args**
        |   None

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # register subclasses by their lowercase name
        name = cls.__name__.lower()
        cls.registry[name] = cls

    def __init__(self, dt: float, sys: SemiLinearFirstOrderSystem,
                 implicit_solver: str) -> None:
        """Initialize the superclasses and attributes.

        This method initializes the ``ABC`` and ``Module`` superclasses and the
        attributes.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear
                first-order O.D.E. system
        |   ``implicit_solver`` (``str``): the implicit solver string key

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclasses and attributes
        super().__init__()
        self.dt = dt
        self.sys = sys
        self.init_implicit_solvers(LinearSolver.lookup(implicit_solver))
        if norm(self.sys.A - diag(diag(self.sys.A)), ord='fro') == 0.0:
            self._diag_A = diag(sys.A)
            self._linear = lambda x: x * self._diag_A
        else:
            self._linear = lambda x: x @ self.sys.A.T


class RK2CN(SemiLinearTimeStepMethod):
    """Represent the 2nd-order Runge-Kutta Crank-Nicolson time-stepping method.

    This class represents the 2nd-order Runge-Kutta Crank-Nicolson
    time-stepping method. See [1] for more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["SemiLinearTimeStepMethod"]]``):
            a registry of semi-linear time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear first-order
            O.D.E. system

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available semi-linear
            time-stepping methods
    |   ``lookup()``: return the subclass whose name matches the input
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``init_implicit_solvers()``: initialize the implicit solvers
    |   ``approx_flow_map()``: the R.K.2.C.N. method ``self.dt`` approximate
            flow map
    |   ``__init__()``: initialize the superclass

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 148.
    """

    def init_implicit_solvers(self, implicit_solver_cls: Type["LinearSolver"]
                              ) -> None:
        """Initialize the implicit solvers.

        This method initializes the implicit solvers. In particular, this
        method initializes the attribute ``self.implicit_solver_insts``, which
        is a ``ModuleList`` of ``LinearSolver`` s used in
        ``self.approx_flow_map()``. See [1] for more information.

        | **Args**
        |   ``implicit_solver_cls`` (``Type["LinearSolver"]``): the implicit
                solver class

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 148.
        """
        # gather the matrices used in the implicit solvers
        Ms = [eye(self.sys.num_states, device=self.sys.A.device)
              - 0.5 * self.dt * self.sys.A]
        # initialize the implicit solvers
        implicit_solver_insts = [implicit_solver_cls(M) for M in Ms]
        self.implicit_solver_insts = ModuleList(implicit_solver_insts)

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """Return the R.K.2.C.N. method ``self.dt`` approximate flow map.

        This method returns the R.K.2.C.N. method ``self.dt`` approximate flow
        map. See [1] for more information.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 148.
        """
        # use the R.K.2.C.N. method to approx. advance the state by one
        # time-step
        rhs_linear = x + 0.5 * self.dt * self._linear(x)
        Nx = self.sys.nonlinear(x)
        rhs1 = rhs_linear + self.dt * Nx
        x1 = self.implicit_solver_insts[0](rhs1)
        rhs2 = rhs_linear + 0.5 * self.dt * (Nx + self.sys.nonlinear(x1))
        x2 = self.implicit_solver_insts[0](rhs2)
        return x2

    def __init__(self, dt: float, sys: SemiLinearFirstOrderSystem,
                 implicit_solver: str) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``SemiLinearTimeStepMethod``.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear
                first-order O.D.E. system
        |   ``implicit_solver`` (``str``): the implicit solver string key

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__(dt, sys, implicit_solver)


class RK3CN(SemiLinearTimeStepMethod):
    """Represent the 3rd-order Runge-Kutta Crank-Nicolson time-stepping method.

    This class represents the 3rd-order Runge-Kutta Crank-Nicolson
    time-stepping method. See [1] for more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["SemiLinearTimeStepMethod"]]``):
            a registry of semi-linear time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear first-order
            O.D.E. system

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available semi-linear
            time-stepping methods
    |   ``lookup()``: return the subclass whose name matches the input
            string key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``init_implicit_solvers()``: initialize the implicit solvers
    |   ``approx_flow_map()``: the R.K.3.C.N. method ``self.dt`` approximate
            flow map
    |   ``__init__()``: initialize the superclass

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 149.
"""

    def init_implicit_solvers(self, implicit_solver_cls: Type["LinearSolver"]
                              ) -> None:
        """Initialize the implicit solvers.

        This method initializes the implicit solvers. In particular, this
        method initializes the attribute ``self.implicit_solver_insts``, which
        is a ``ModuleList`` of ``LinearSolver`` s used in
        ``self.approx_flow_map()``. See [1] for more information.

        | **Args**
        |   ``implicit_solver_cls`` (``Type["LinearSolver"]``): the implicit
                solver class

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 149.
        """
        Bp = [1.0 / 6, 5.0 / 24, 1.0 / 8]
        Ms = [eye(self.sys.num_states, device=self.sys.A.device)
              - b * self.dt * self.sys.A for b in Bp]
        implicit_solver_insts = [implicit_solver_cls(M) for M in Ms]
        self.implicit_solver_insts = ModuleList(implicit_solver_insts)

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """Return the R.K.3.C.N. method ``self.dt`` approximate flow map.

        Return the R.K.3.C.N. method ``self.dt`` approximate flow map. See [1]
        for more information.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (sys.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (sys.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 149.
        """
        A = [0, -5.0 / 9, -153.0 / 128]
        B = [1.0 / 3, 15.0 / 16, 8.0 / 15]
        Bp = [1.0 / 6, 5.0 / 24, 1.0 / 8]
        Q1 = self.dt * self.sys.nonlinear(x)
        rhs1 = x + B[0] * Q1 + Bp[0] * self.dt * self._linear(x)
        x1 = self.implicit_solver_insts[0](rhs1)
        Q2 = A[1] * Q1 + self.dt * self.sys.nonlinear(x1)
        rhs2 = x1 + B[1] * Q2 + Bp[1] * self.dt * self._linear(x1)
        x2 = self.implicit_solver_insts[1](rhs2)
        Q3 = A[2] * Q2 + self.dt * self.sys.nonlinear(x2)
        rhs3 = x2 + B[2] * Q3 + Bp[2] * self.dt * self._linear(x2)
        x3 = self.implicit_solver_insts[2](rhs3)
        return x3

    def __init__(self, dt: float, sys: SemiLinearFirstOrderSystem,
                 implicit_solver: str) -> None:
        """Initialize the superclass.

        This method initializes the superclass ``SemiLinearTimeStepMethod``.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear
                first-order O.D.E. system
        |   ``implicit_solver`` (``str``): the implicit solver string key

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the superclass
        super().__init__(dt, sys, implicit_solver)


# utilize various time-stepping methods for time discretization
class DiscretizedFirstOrderSystem(DiscreteSystem):
    """Represent a discretized first-order O.D.E. system.

    This class represents a discretized first-order O.D.E. system.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   None

    | **Attributes**
    |   ``field`` (``str``): the field the dynamical system is defined over
    |   ``num_states`` (``int``): the number of states
    |   ``method`` (``Union[ExplicitTimeStepMethod,
            SemiLinearTimeStepMethod]``): the time-stepping method

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   None

    | **Methods**
    |   ``map()``: return the output of ``self.method.approx_flow_map()``
    |   ``__init__()``: initialize the superclass and attributes

    | **References**
    |   None
    """

    @property
    def field(self) -> str:
        return self.method.sys.field

    @property
    def num_states(self) -> int:
        return self.method.sys.num_states

    def map(self, x: Tensor) -> Tensor:
        """Return the output of ``self.method.approx_flow_map()``.

        This method returns the output of ``self.method.approx_flow_map()``.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (self.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (self.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   None
        """
        return self.method.approx_flow_map(x)

    def __init__(self, method: Union[ExplicitTimeStepMethod,
                                     SemiLinearTimeStepMethod]) -> None:
        """Initialize the superclass and attributes.

        This method initialize the superclass and attributes.

        | **Args**
        |   ``method`` (``Union[ExplicitTimeStepMethod,
                SemiLinearTimeStepMethod]``): the time-stepping method

        | **Return**
        |   None

        | **Raises**
        |   None

        | **References**
        |   None
        """
        # initialize the time-stepping method and the first-order O.D.E. system
        super().__init__()
        self.method = method


def gen_approx_discrete_sys(sys: FirstOrderSystem, dt: float, method: str,
                            implicit_solver: Optional[str] = None
                            ) -> DiscreteSystem:
    """Return an approximate discrete system from a first-order O.D.E. system.

    This method returns an approximate discrete system from a first-order
    O.D.E. system. The approximate discrete system is generated using a
    time-stepping method. The time-stepping method is specified by the
    input string key. If the solver is implicit, then the implicit solver
    is specified by the input string key.

    | **Args**
    |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system
    |   ``dt`` (``float``): the time-step
    |   ``method`` (``str``): the time-stepping method string key
    |   ``implicit_solver`` (``Optional[str]``): the implicit solver string
            key

    | **Return**
    |   ``DiscreteSystem``: the approximate discrete system

    | **Raises**
    |   ``ValueError``: if the time-stepping method is semi-linear and the
            first-order O.D.E. system is not semi-linear
    |   ``ValueError``: if the time-stepping method is semi-linear and the
            implicit solver is not specified

    | **References**
    |   None
    """
    try:
        # see if the desired time-stepping method is semi-linear
        method_cls_semi = SemiLinearTimeStepMethod.lookup(method)
        # see if the first-order O.D.E. system is semi-linear
        if not isinstance(sys, SemiLinearFirstOrderSystem):
            raise ValueError("Semi-linear time-stepping methods require a "
                             + "semi-linear first-order O.D.E. system")
        # see if the implicit solver is specified
        if implicit_solver not in LinearSolver.methods():
            raise ValueError("Semi-linear time-stepping methods require a "
                             + "semi-linear implicit solver")
        # return the approximate discrete system
        method_inst_semi = method_cls_semi(dt, sys, implicit_solver)
        return DiscretizedFirstOrderSystem(method_inst_semi)
    except NotImplementedError:
        # see if the desired time-stepping method is explicit
        method_cls_exp = ExplicitTimeStepMethod.lookup(method)
        # return the approximate discrete system
        method_inst_exp = method_cls_exp(dt, sys)
        return DiscretizedFirstOrderSystem(method_inst_exp)
