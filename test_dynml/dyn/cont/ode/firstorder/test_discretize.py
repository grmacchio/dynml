"""Test the ``dynml.dyn.cont.ode.firstorder.discretize`` module.

This module tests the ``dynml.dyn.cont.ode.firstorder.discretize`` module.
"""


# import built-in python-package code
from typing import Type
# import external python-package code
from torch import diag, eye, float64, set_default_dtype, tensor, Tensor
from torch.cuda import is_available
from torch.linalg import lu_factor
from torch.nn import Parameter
# import internal python-package code
from dynml.dyn.cont.ode.firstorder.discretize import ExplicitTimeStepMethod
from dynml.dyn.cont.ode.firstorder.discretize import Euler, RK2, RK4
from dynml.dyn.cont.ode.firstorder.discretize import LinearSolver, LU, Inv
from dynml.dyn.cont.ode.firstorder.discretize import Diag
from dynml.dyn.cont.ode.firstorder.discretize import SemiLinearTimeStepMethod
from dynml.dyn.cont.ode.firstorder.discretize import RK2CN, RK3CN
from dynml.dyn.cont.ode.firstorder.discretize import DiscretizedFirstOrderSystem  # noqa: E501
from dynml.dyn.cont.ode.firstorder.discretize import gen_approx_discrete_sys
from dynml.dyn.cont.ode.firstorder.system import FirstOrderSystem
from dynml.dyn.cont.ode.firstorder.system import SemiLinearFirstOrderSystem
from test_dynml.dyn.cont.ode.firstorder.test_system import FirstOrderSystemExample            # noqa: E501
from test_dynml.dyn.cont.ode.firstorder.test_system import SemiLinearFirstOrderSystemExample  # noqa: E501


# test ExplicitTimeStepMethod
class ExplicitTimeStepMethodExample(ExplicitTimeStepMethod):
    """Represent a subclass of ``ExplicitTimeStepMethod``.

    This class represents a subclass of ``ExplicitTimeStepMethod``.

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
    |   ``lookup()``: return the subclass whose name matches the input
            string key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``approx_flow_map()``: the Euler's method ``self.dt`` approximate flow
            map
    |   ``__init__()``: initialize the time-step and the first-order O.D.E.

    | **References**
    |   None
    """

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The Euler's method ``self.dt`` approximate flow map.

        This method is the Euler's method ``self.dt`` approximate flow map

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
        # use Euler's method to approx. advance the state by one time-step
        return x + self.dt * self.sys.rhs(x)

    def __init__(self, dt: float, sys: FirstOrderSystem) -> None:
        """Initialize the time-step and the first-order O.D.E.

        This method initializes the time-step and the first-order O.D.E.

        | **Args**
        |   ``dt`` (``float``): the time-step
        |   ``sys`` (``FirstOrderSystem``): the first-order O.D.E. system

        | **Return**
        |  None

        | **Raises**
        |  None

        | **References**
        |  None
        """
        # call the superclass constructor
        super().__init__(dt, sys)


def test_ExplicitTimeStepMethod() -> None:
    """Test the ``ExplicitTimeStepMethod`` class.

    This method tests the ``ExplicitTimeStepMethod`` class. In particular,
    this method instantiates ``ExplicitTimeStepMethodExample`` and tests
    ``registry``, ``dt``, ``sys``, ``methods()``, ``lookup()``, and
    ``approx_flow_map()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of ExplicitTimeStepMethodExample
    dt = 1.0
    sys = FirstOrderSystemExample().to(device)
    test = ExplicitTimeStepMethodExample(dt, sys)
    # test registry
    assert ("ExplicitTimeStepMethodExample".lower()
            in ExplicitTimeStepMethod.registry.keys())
    # test dt
    assert test.dt == dt
    # test sys
    assert test.sys == sys
    # test methods()
    assert ("ExplicitTimeStepMethodExample".lower()
            in ExplicitTimeStepMethod.methods())
    # test lookup()
    assert (ExplicitTimeStepMethod.lookup("ExplicitTimeStepMethodExample")
            == ExplicitTimeStepMethodExample)
    try:
        ExplicitTimeStepMethod.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown time-stepping method")
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = x + dt * sys.rhs(x)
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)


# test Euler
def test_Euler() -> None:
    """Test the ``Euler`` class.

    This method tests the ``Euler`` class. In particular, this method
    instantiates an ``Euler`` object and tests ``registry``, ``dt``, ``sys``,
    ``approx_flow_map()``, ``methods()``, and ``lookup()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of Euler
    dt = 1.0
    sys = FirstOrderSystemExample().to(device)
    test = Euler(dt, sys)
    # test registry
    assert ("euler" in Euler.registry.keys())
    # test dt
    assert test.dt == dt
    # test sys
    assert test.sys == sys
    # test methods()
    assert ("euler" in Euler.methods())
    # test lookup()
    assert Euler.lookup("euler") == Euler
    try:
        Euler.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown time-stepping method")
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = x + dt * sys.rhs(x)
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)


# test RK2
def test_RK2() -> None:
    """Test the ``RK2`` class.

    This method tests the ``RK2`` class. In particular, this method
    instantiates an ``RK2`` object and tests ``registry``, ``dt``, ``sys``,
    ``approx_flow_map()``, ``methods()``, and ``lookup()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of RK2
    dt = 1.0
    sys = FirstOrderSystemExample().to(device)
    test = RK2(dt, sys)
    # test registry
    assert ("rk2" in RK2.registry.keys())
    # test dt
    assert test.dt == dt
    # test methods()
    assert ("rk2" in RK2.methods())
    # test lookup()
    assert RK2.lookup("rk2") == RK2
    try:
        RK2.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown time-stepping method")
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    u1 = x + 0.5 * dt * sys.rhs(x)
    desired = x + dt * sys.rhs(u1)
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)


# test RK4
def test_RK4() -> None:
    """Test the ``RK4`` class.

    This method tests the ``RK4`` class. In particular, this method
    instantiates an ``RK4`` object and tests ``registry``, ``dt``, ``sys``,
    ``approx_flow_map()``, ``methods()``, and ``lookup()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of RK4
    dt = 1.0
    sys = FirstOrderSystemExample().to(device)
    test = RK4(dt, sys)
    # test registry
    assert ("rk4" in RK4.registry.keys())
    # test dt
    assert test.dt == dt
    # test methods()
    assert ("rk4" in RK4.methods())
    # test lookup()
    assert RK4.lookup("rk4") == RK4
    try:
        RK4.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown time-stepping method")
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    k1 = sys.rhs(x)
    k2 = sys.rhs(x + 0.5 * dt * k1)
    k3 = sys.rhs(x + 0.5 * dt * k2)
    k4 = sys.rhs(x + dt * k3)
    desired = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)


# test LinearSolver
class LinearSolverExample(LinearSolver):
    """Represent a subclass of ``LinearSolver``.

    This class represents a subclass of ``LinearSolver``. Here, we are
    interested in solving the linear system ``xM^T = y`` using the matrix
    inverse.

    | **Abstract Attributes**
    |  None

    | **Class Attributes**
    |  ``registry`` (``Dict[str, Type["LinearSolver"]]``): a registry of
            linear solvers

    | **Attributes**
    |  ``M_inv`` (``Tensor``): the inverse of the input matrix

    | **Abstract Methods**
    |  None

    | **Class Methods**
    |  ``methods()``: return the list of available linear solvers
    |  ``lookup()``: return the subclass whose name matches the input string
            key
    |  ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |  ``__init__()``: compute the matrix inverse and initialize the superclass
    |  ``__call__()``: return the solution ``x`` given the input ``y``

    | **References**
    |  None
    """

    def __init__(self, M: Tensor) -> None:
        """Compute the matrix inverse and initialize the superclass.

        This method computes the matrix inverse and initializes the superclass.

        | **Args**
        |   ``M`` (``Tensor``): the matrix ``M`` in the linear system with
                shape ``(n, n)``

        | **Return**
        |  None

        | **Raises**
        |  None

        | **References**
        |  None
        """
        # compute the matrix inverse and initialize the superclass
        super().__init__(M)
        self.M_inv = Parameter(M.inverse(), requires_grad=False)

    def __call__(self, y: Tensor) -> Tensor:
        """Return the solution ``x`` given the input ``y``.

        This method returns the solution ``x`` given the input ``y``. Here, the
        input and solution satisfy the linear system of equations:
        ``xM^T = y``.

        | **Args**
        |   ``y`` (``Tensor``): the input ``y`` with shape
                ``(...,) + (n,)``

        | **Return**
        |  ``Tensor``: the solution ``x`` with shape ``(...,) + (n,)``

        | **Raises**
        |  None

        | **References**
        |  None
        """
        return y @ self.M_inv.t()


def test_LinearSolver() -> None:
    """Test the ``LinearSolver`` class.

    This method tests the ``LinearSolver`` class. In particular, this method
    instantiates ``LinearSolverExample`` and tests ``registry``, ``M_inv``,
    ``__call__()``, ``methods()``, ``lookup()``, finally we test if ``.to()``
    capabilities are inherited from ``torch.nn.Module``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of LinearSolverExample
    M = tensor([[2.0, 0.0], [0.0, 3.0]], device=device)
    M_inv = tensor([[1 / 2.0, 0.0], [0.0, 1.0 / 3.0]], device=device)
    test = LinearSolverExample(M)
    # test registry
    assert ("LinearSolverExample".lower() in LinearSolver.registry.keys())
    # test M_inv
    assert test.M_inv.allclose(M_inv, atol=0.0)
    # test __call__()
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    desired = y @ M_inv
    assert test(y).allclose(desired, atol=0.0)
    # test methods()
    assert ("LinearSolverExample".lower() in LinearSolver.methods())
    # test lookup()
    assert (LinearSolver.lookup("LinearSolverExample")
            == LinearSolverExample)
    try:
        LinearSolver.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown linear solver")
    # test .to() capabilities
    test = test.to('cpu')
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device='cpu')
    M_inv = M_inv.to('cpu')
    desired = y @ M_inv
    assert test(y).allclose(desired, atol=0.0)


# test LU
def test_LU() -> None:
    """Test the ``LU`` class.

    This method tests the ``LU`` class. In particular, this method instantiates
    an ``LU`` object and tests ``registry``, ``LU``, ``pivots``,
    ``__call()__``, ``methods()``, ``lookup()``, and finally we test if
    ``.to()`` capabilities are inherited from ``torch.nn.Module``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of LU
    M = tensor([[2.0, 0.0], [0.0, 3.0]], device=device)
    M_inv = tensor([[1 / 2.0, 0.0], [0.0, 1.0 / 3.0]], device=device)
    test = LU(M)
    # test registry
    assert ("lu" in LU.registry.keys())
    # test LU
    _LU, _pivots = lu_factor(M)
    assert test.LU.allclose(_LU, atol=0.0)
    # test pivots
    assert test.pivots.allclose(_pivots, atol=0.0)
    # test __call__()
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    desired = y @ M_inv
    assert test(y).allclose(desired, atol=0.0)
    # test methods()
    assert ("lu" in LU.methods())
    # test lookup()
    assert LU.lookup("lu") == LU
    try:
        LU.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown linear solver")
    # test .to() capabilities
    test = test.to('cpu')
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device='cpu')
    M_inv = M_inv.to('cpu')
    desired = y @ M_inv
    assert test(y).allclose(desired, atol=0.0)


def test_Inv() -> None:
    """Test the ``Inv`` class.

    This method tests the ``Inv`` class. In particular, this method
    instantiates an ``Inv`` object and tests ``registry``, ``M_inv``,
    ``__call__()``, ``methods()``, ``lookup()``, and finally we test if
    ``.to()`` capabilities are inherited from ``torch.nn.Module``.

    | **Args**
    |   None

    | **Return**
    |  None

    | **Raises**
    |  None

    | **References**
    |  None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of Inv
    M = tensor([[2.0, 1.0], [1.0, 2.0]], device=device)
    M_inv = tensor([[2.0, -1.0], [-1.0, 2.0]], device=device) / 3.0
    test = Inv(M)
    # test registry
    assert ("inv" in Inv.registry.keys())
    # test M_inv
    assert test.M_inv.allclose(M_inv, atol=0.0)
    # test __call__()
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    desired = y @ M_inv.T
    assert test(y).allclose(desired, atol=0.0)
    # test methods()
    assert ("inv" in Inv.methods())
    # test lookup()
    assert Inv.lookup("inv") == Inv
    try:
        Inv.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown linear solver")
    # test .to() capabilities
    test = test.to('cpu')
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device='cpu')
    M_inv = M_inv.to('cpu')
    desired = y @ M_inv.T
    assert test(y).allclose(desired, atol=0.0)


# test Diag
def test_Diag() -> None:
    """Test the ``Diag`` class.

    This method tests the ``Diag`` class. In particular, this method
    instantiates a ``Diag`` object and tests ``registry``, ``d_inv``,
    ``__call__()``, ``methods()``, ``lookup()``, and finally we test if
    ``.to()`` capabilities are inherited from ``torch.nn.Module``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of Diag
    M = tensor([[2.0, 0.0], [0.0, 3.0]], device=device)
    M_inv = tensor([[1 / 2.0, 0.0], [0.0, 1.0 / 3.0]], device=device)
    test = Diag(M)
    # test registry
    assert ("diag" in Diag.registry.keys())
    # test d_inv
    assert test.d_inv.allclose(diag(M_inv), atol=0.0)
    # test __call__()
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device=device)
    desired = y @ M_inv.T
    assert test(y).allclose(desired, atol=0.0)
    # test methods()
    assert ("diag" in Diag.methods())
    # test lookup()
    assert Diag.lookup("diag") == Diag
    try:
        Diag.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown linear solver")
    # test .to() capabilities
    test = test.to('cpu')
    y = tensor([[5.0, 6.0], [7.0, 8.0]], device='cpu')
    M_inv = M_inv.to('cpu')
    desired = y @ M_inv.T
    assert test(y).allclose(desired, atol=0.0)


# test SemiLinearTimeStepMethod
class SemiLinearTimeStepMethodExample(SemiLinearTimeStepMethod):
    """Represent a subclass of ``SemiLinearTimeStepMethod``.

    This class represents a subclass of ``SemiLinearTimeStepMethod``. Here,
    we are will be using the 2nd-order Runge-Kutta Crank-Nicolson (R.K.2.C.N.)
    method. See [1] for more information.

    | **Abstract Attributes**
    |   None

    | **Class Attributes**
    |   ``registry`` (``Dict[str, Type["SemiLinearTimeStepMethod"]]``): a
            registry of semi-linear time-stepping methods

    | **Attributes**
    |   ``dt`` (``float``): the time-step
    |   ``sys`` (``SemiLinearFirstOrderSystem``): the semi-linear first-order
            O.D.E. system

    | **Abstract Methods**
    |   None

    | **Class Methods**
    |   ``methods()``: return the list of available semi-linear time-stepping
            methods
    |   ``lookup()``: return the subclass whose name matches the input
            string key
    |   ``__init_subclass__()``: register subclasses by their lowercase name

    | **Methods**
    |   ``init_implicit_solvers()``: initialize the implicit solvers
    |   ``approx_flow_map()``: the R.K.2.C.N. method ``self.dt`` approximate
            flow map
    |   ``__init__()``: initialize the time-step, the semi-linear first-order
            O.D.E., and the implicit solvers

    | **References**
    |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous Flow.
            Vol. 148. New York: Springer, 2002. p. 148.
    """

    def init_implicit_solvers(self, implicit_solver_cls: Type["LinearSolver"]
                              ) -> None:
        """Initialize the implicit solvers.

        This method initializes the implicit solvers. See [1] for more
        information.

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
        # initialize the implicit solvers
        self.Ms = [eye(self.sys.num_states) - 0.5 * self.dt * self.sys.A]
        self.implicit_solvers_insts = [implicit_solver_cls(M) for M in self.Ms]

    def approx_flow_map(self, x: Tensor) -> Tensor:
        """The R.K.2.C.N. method ``self.dt`` approximate flow map.

        This method is the R.K.2.C.N. method ``self.dt`` approximate flow map.
        See [1] for more information.

        | **Args**
        |   ``x`` (``Tensor``): the input state with shape
                ``(...,) + (self.num_states,)``

        | **Return**
        |   ``Tensor``: approximately, the input state advanced by one
                time-step with shape ``(...,) + (self.num_states,)``

        | **Raises**
        |   None

        | **References**
        |   [1] Peyret, Roger. Spectral Methods for Incompressible Viscous
                Flow. Vol. 148. New York: Springer, 2002. p. 148.
        """
        # use the R.K.2.C.N. method to approx. advance the state by one
        # time-step
        un = x
        rhs1 = (un + self.dt * self.sys.nonlinear(un)
                + 0.5 * self.dt * un @ self.sys.A.T)
        u1 = self.implicit_solvers_insts[0](rhs1)
        rhs2 = un + 0.5 * self.dt * (self.sys.nonlinear(un)
                                     + self.sys.nonlinear(u1)
                                     + un @ self.sys.A.T)
        return self.implicit_solvers_insts[0](rhs2)

    def __init__(self, dt: float, sys: SemiLinearFirstOrderSystem,
                 implicit_solver: str) -> None:
        """Initialize the time-step, the semi-linear first-order O.D.E., and
        the implicit solvers.

        This method initializes the time-step, the semi-linear first-order
        O.D.E., and the implicit solvers.

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
        # initialize the time-step, the semi-linear first-order O.D.E., and the
        # implicit solvers
        super().__init__(dt, sys, implicit_solver)


def test_SemiLinearTimeStepMethod() -> None:
    """Test the ``SemiLinearTimeStepMethod`` class

    This method tests the ``SemiLinearTimeStepMethod`` class. In particular,
    this method instantiates ``SemiLinearTimeStepMethodExample`` and tests
    ``registry``, ``dt``, ``sys``, ``init_implicit_solvers()``,
    ``approx_flow_map()``, ``methods()``, and ``lookup()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of SemiLinearTimeStepMethodExample
    dt = 1.0
    sys = SemiLinearFirstOrderSystemExample()
    test = SemiLinearTimeStepMethodExample(dt, sys, 'lu').to(device)
    # test registry
    assert ("SemiLinearTimeStepMethodExample".lower()
            in SemiLinearTimeStepMethod.registry.keys())
    # test dt
    assert test.dt == dt
    # test sys
    assert test.sys == sys
    # test init_implicit_solvers()
    Id = eye(sys.num_states).to(device)
    implicit_solver = LU(Id - 0.5 * dt * sys.A)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = implicit_solver(x)
    assert test.implicit_solvers_insts[0](x).allclose(desired, atol=0.0)
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    u1 = implicit_solver(x + dt * sys.nonlinear(x) + 0.5 * dt * x @ sys.A.T)
    desired = implicit_solver(x + 0.5 * dt * (sys.nonlinear(x)
                                              + sys.nonlinear(u1)
                                              + x @ sys.A.T))
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)
    # test methods()
    assert ("SemiLinearTimeStepMethodExample".lower()
            in SemiLinearTimeStepMethod.methods())
    # test lookup()
    assert (SemiLinearTimeStepMethod.lookup("SemiLinearTimeStepMethodExample")
            == SemiLinearTimeStepMethodExample)
    try:
        SemiLinearTimeStepMethod.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown semi-linear "
                            + "time-stepping method")


# test RK2CN
def test_RK2CN() -> None:
    """Test the ``RK2CN`` class.

    This method tests the ``RK2CN`` class. In particular, this method
    instantiates an ``RK2CN`` object and tests ``registry``, ``dt``, ``sys``,
    ``init_implicit_solvers()``, ``approx_flow_map()``, ``methods()``, and
    ``lookup()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of RK2CN
    dt = 1.0
    sys = SemiLinearFirstOrderSystemExample()
    test = RK2CN(dt, sys, 'lu').to(device)
    # test registry
    assert ("rk2cn" in RK2CN.registry.keys())
    # test dt
    assert test.dt == dt
    # test sys
    assert test.sys == sys
    # test init_implicit_solvers()
    Id = eye(sys.num_states).to(device)
    implicit_solver = LU(Id - 0.5 * dt * sys.A)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = implicit_solver(x)
    assert test.implicit_solver_insts[0](x).allclose(desired, atol=0.0)
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    rk2cn = SemiLinearTimeStepMethodExample(dt, sys, 'lu')
    desired = rk2cn.approx_flow_map(x)
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)
    # test methods()
    assert ("rk2cn" in RK2CN.methods())
    # test lookup()
    assert RK2CN.lookup("rk2cn") == RK2CN
    try:
        RK2CN.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown semi-linear "
                            + "time-stepping method")


# test RK3CN
def test_RK3CN() -> None:
    """Test the ``RK3CN`` class.

    This method tests the ``RK3CN`` class. In particular, this method
    instantiates an ``RK3CN`` object and tests ``registry``, ``dt``, ``sys``,
    ``init_implicit_solvers()``, ``approx_flow_map()``, ``methods()``, and
    ``lookup()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of RK3CN
    dt = 1.0
    sys = SemiLinearFirstOrderSystemExample()
    test = RK3CN(dt, sys, 'lu').to(device)
    # test registry
    assert ("rk3cn" in RK3CN.registry.keys())
    # test dt
    assert test.dt == dt
    # test sys
    assert test.sys == sys
    # test init_implicit_solvers()
    Id = eye(sys.num_states).to(device)
    implicit_solver1 = LU(Id - 1/6 * dt * sys.A)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = implicit_solver1(x)
    assert test.implicit_solver_insts[0](x).allclose(desired, atol=0.0)
    Id = eye(sys.num_states).to(device)
    implicit_solver2 = LU(Id - 5/24 * dt * sys.A)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = implicit_solver2(x)
    assert test.implicit_solver_insts[1](x).allclose(desired, atol=0.0)
    Id = eye(sys.num_states).to(device)
    implicit_solver3 = LU(Id - 1/8 * dt * sys.A)
    x = tensor([[0.0, 1.0], [2.0, 3.0]])
    desired = implicit_solver3(x)
    assert test.implicit_solver_insts[2](x).allclose(desired, atol=0.0)
    # test approx_flow_map()
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    u0 = x
    Q1 = 0 + dt * sys.nonlinear(u0)
    u1 = implicit_solver1(u0 + 1/3 * Q1 + 1/6 * dt * u0 @ sys.A.T)
    Q2 = -5/9 * Q1 + dt * sys.nonlinear(u1)
    u2 = implicit_solver2(u1 + 15/16 * Q2 + 5/24 * dt * u1 @ sys.A.T)
    Q3 = -153/128 * Q2 + dt * sys.nonlinear(u2)
    desired = implicit_solver3(u2 + 8/15 * Q3 + 1/8 * dt * u2 @ sys.A.T)
    assert test.approx_flow_map(x).allclose(desired, atol=0.0)
    # test methods()
    assert ("rk3cn" in RK3CN.methods())
    # test lookup()
    assert RK3CN.lookup("rk3cn") == RK3CN
    try:
        RK3CN.lookup("NotAKey")
        raise NotImplementedError("Test not implemented")
    except NotImplementedError as exc:
        assert str(exc) == ("'NotAKey' is an unknown semi-linear "
                            + "time-stepping method")


# test DiscretizedFirstOrderSystem
def test_DiscretizedFirstOrderSystem() -> None:
    """Test the ``DiscretizedFirstOrderSystem`` class.

    This method tests the ``DiscretizedFirstOrderSystem`` class. In particular,
    this method instantiates a ``DiscretizedFirstOrderSystem`` object and tests
    ``num_states``, ``method``, and ``map()``.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # create an instance of DiscretizedFirstOrderSystem
    sys1 = FirstOrderSystemExample()
    method1 = RK2(1.0, sys1)
    test1 = DiscretizedFirstOrderSystem(method1).to(device)
    sys2 = SemiLinearFirstOrderSystemExample()
    method2 = RK2CN(1.0, sys2, 'lu')
    test2 = DiscretizedFirstOrderSystem(method2).to(device)
    # test num_states
    assert test1.num_states == sys1.num_states
    assert test2.num_states == sys2.num_states
    # test method
    assert test1.method == method1
    assert test2.method == method2
    # test map()
    x1 = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired1 = method1.approx_flow_map(x1)
    assert test1.map(x1).allclose(desired1, atol=0.0)
    x2 = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired2 = method2.approx_flow_map(x2)
    assert test2.map(x2).allclose(desired2, atol=0.0)


# test gen_approx_discrete_sys
def test_gen_approx_discrete_sys() -> None:
    """Test the ``gen_approx_discrete_sys`` method.

    This method tests the ``gen_approx_discrete_sys`` method. In particular,
    we test the method's ability to fetch any known time-stepping method and
    generate the approximate discrete system. Next, we test the ``ValueError``
    raised when the time-stepping method is semi-linear and the first-order
    O.D.E. system is not semi-linear. Finally, we test the ``ValueError``
    raised when the time-stepping method is semi-linear and an implicit solver
    is not provided.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # set torch to float64
    set_default_dtype(float64)
    # find the device
    device = 'cuda' if is_available() else 'cpu'
    # generate the approximate discrete system using Euler's method
    fos1 = FirstOrderSystemExample()
    ds1 = gen_approx_discrete_sys(fos1, 1.0, 'euler').to(device)
    fos2 = FirstOrderSystemExample().to(device)
    tsm1 = Euler(1.0, fos2)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = tsm1.approx_flow_map(x)
    assert ds1.map(x).allclose(desired, atol=0.0)
    _fos1 = FirstOrderSystemExample().to(device)
    _ds1 = gen_approx_discrete_sys(_fos1, 1.0, 'euler').to(device)
    _fos2 = FirstOrderSystemExample().to(device)
    _tsm1 = Euler(1.0, _fos2)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = _tsm1.approx_flow_map(x)
    assert _ds1.map(x).allclose(desired, atol=0.0)
    # generate the approximate discrete system using RK2
    ds2 = gen_approx_discrete_sys(fos1, 1.0, 'rk2').to(device)
    tsm2 = RK2(1.0, fos2)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = tsm2.approx_flow_map(x)
    assert ds2.map(x).allclose(desired, atol=0.0)
    _ds2 = gen_approx_discrete_sys(_fos1, 1.0, 'rk2').to(device)
    _tsm2 = RK2(1.0, _fos2)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = _tsm2.approx_flow_map(x)
    assert _ds2.map(x).allclose(desired, atol=0.0)
    # generate the approximate discrete system using RK4
    ds3 = gen_approx_discrete_sys(fos1, 1.0, 'rk4').to(device)
    tsm3 = RK4(1.0, fos2)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = tsm3.approx_flow_map(x)
    assert ds3.map(x).allclose(desired, atol=0.0)
    _ds3 = gen_approx_discrete_sys(_fos1, 1.0, 'rk4').to(device)
    _tsm3 = RK4(1.0, _fos2)
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = _tsm3.approx_flow_map(x)
    assert _ds3.map(x).allclose(desired, atol=0.0)
    # generate the approximate discrete system using RK2CN
    fos3 = SemiLinearFirstOrderSystemExample()
    ds4 = gen_approx_discrete_sys(fos3, 1.0, 'rk2cn', 'lu').to(device)
    fos4 = SemiLinearFirstOrderSystemExample().to(device)
    tsm4 = RK2CN(1.0, fos4, 'lu')
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = tsm4.approx_flow_map(x)
    assert ds4.map(x).allclose(desired, atol=0.0)
    _fos3 = SemiLinearFirstOrderSystemExample()
    _ds4 = gen_approx_discrete_sys(_fos3, 1.0, 'rk2cn', 'lu').to(device)
    _fos4 = SemiLinearFirstOrderSystemExample().to(device)
    _tsm4 = RK2CN(1.0, _fos4, 'lu')
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = _tsm4.approx_flow_map(x)
    assert _ds4.map(x).allclose(desired, atol=0.0)
    # generate the approximate discrete system using RK3CN
    ds5 = gen_approx_discrete_sys(fos3, 1.0, 'rk3cn', 'lu').to(device)
    tsm5 = RK3CN(1.0, fos4, 'lu')
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = tsm5.approx_flow_map(x)
    assert ds5.map(x).allclose(desired, atol=0.0)
    _ds5 = gen_approx_discrete_sys(_fos3, 1.0, 'rk3cn', 'lu').to(device)
    _tsm5 = RK3CN(1.0, _fos4, 'lu')
    x = tensor([[0.0, 1.0], [2.0, 3.0]], device=device)
    desired = _tsm5.approx_flow_map(x)
    assert _ds5.map(x).allclose(desired, atol=0.0)
    # test ValueError when the time-stepping method is semi-linear and the
    # first-order O.D.E. system is not semi-linear
    try:
        fos5 = FirstOrderSystemExample()
        gen_approx_discrete_sys(fos5, 1.0, 'rk2cn', 'lu')
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == ("Semi-linear time-stepping methods require a "
                            + "semi-linear first-order O.D.E. system")
    # test ValueError when the time-stepping method is semi-linear and an
    # implicit solver is not provided
    try:
        fos6 = SemiLinearFirstOrderSystemExample()
        gen_approx_discrete_sys(fos6, 1.0, 'rk2cn')
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == ("Semi-linear time-stepping methods require a "
                            + "semi-linear implicit solver")
