"""Test the dynml.dyn.discrete.numsolve module.

This module tests the dynml.dyn.discrete.numsolve module.
"""

# import built-in python-package code
from math import prod
# import external python-package code
from torch import mean, no_grad, ones_like, randn
from torch import stack, Tensor
from torch.cuda import is_available, mem_get_info
# import internal python-package code
from dynml.dyn.discrete.numsolve import gen_num_trajs
from dynml.utils.config import config
from test_dynml.dyn.discrete.test_system import DiscreteSystemExample


# test gen_num_trajs
def test_gen_num_trajs() -> None:  # noqa: C901
    """Test the ``gen_num_trajs()`` method.

    This method tests the ``gen_num_trajs()`` method. In particular, it checks
    the error raised when the compute device is not recognized, it checks the
    error raised when the output device is not recognized, it checks the error
    raised when the initial condition's number of states does not match the
    discrete system's number of states, and it tests the method's ability to
    generate the numerical trajectories for a system whose map is defined by
    adding one to the previous state. Furthermore, this method tests the
    method's ability to compute on either the C.P.U. or G.P.U. and output to
    either the C.P.U. or G.P.U. Finally, this method tests the method's ability
    to compute on the G.P.U. and output to the C.P.U. for a dataset larger than
    the G.P.U.'s available memory.

    | **Args**
    |   None

    | **Return**
    |   None

    | **Raises**
    |   None

    | **References**
    |   None
    """
    # test that an error is raised when the compute device is not recognized
    config(64, 0)
    try:
        dims_state = (4, 3)
        num_traj = 4
        num_samples = 4
        ds = DiscreteSystemExample(dims_state, True)
        def gen_ic_1() -> Tensor:  # noqa: E306
            return randn(num_states,)
        gen_num_trajs(ds, gen_ic_1, num_traj, num_samples, compute='unknown')
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == "The compute device is not recognized"
    # test that an error is raised when the output device is not recognized
    config(64, 0)
    try:
        dims_state = (4, 3)
        num_traj = 4
        num_samples = 4
        ds = DiscreteSystemExample(dims_state, True)
        def gen_ic_2() -> Tensor:  # noqa: E306
            return randn(num_states,)
        gen_num_trajs(ds, gen_ic_2, num_traj, num_samples, output='unknown')
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == "The output device is not recognized"
    # test that an error is raised when the initial condition's number of
    # states does not match the discrete system's number of states
    config(64, 0)
    try:
        dims_state = (4, 3)
        num_traj = 4
        num_samples = 4
        ds = DiscreteSystemExample(dims_state, True)
        def gen_ic_incorrect() -> Tensor:  # noqa: E306
            return randn((4, 2))
        gen_num_trajs(ds, gen_ic_incorrect, num_traj, num_samples)
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == ("The initial condition's state dimensions does "
                            + "not match the discrete system's state "
                            + "dimensions")
    # test gen_num_trajs() computing on C.P.U. and output to the C.P.U. and
    # backwards differentiation with respect to gen_num_trajs() output
    config(64, 0)
    dims_state = (4, 3)
    num_traj = 4
    num_samples = 4
    compute = 'cpu'
    output = 'cpu'
    ds = DiscreteSystemExample(dims_state, True)
    def gen_ic() -> Tensor:  # noqa: E306
        return randn(dims_state,)
    test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                         compute=compute, output=output, pbar=False)
    assert test.device.type == output
    assert tuple(test.shape) == (num_traj, num_samples) + dims_state
    config(64, 0)
    desired = stack(tuple(gen_ic() for _ in range(num_traj)), dim=0).to(output)
    for k in range(num_samples):
        assert test[:, k].allclose(desired, atol=0.0)
        desired = desired + 1.0
    mean(test - ones_like(test, device=output)).backward()
    # test gen_num_trajs() computing on G.P.U. and output to the G.P.U.
    if is_available():
        config(64, 0)
        dims_state = (4, 3)
        num_traj = 4
        num_samples = 4
        compute = 'cuda'
        output = 'cuda'
        ds = DiscreteSystemExample(dims_state, True)
        def gen_ic() -> Tensor:  # noqa: E306
            return randn(dims_state)
        test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                             compute=compute, output=output, pbar=False)
        assert test.device.type == output
        assert tuple(test.shape) == (num_traj, num_samples) + dims_state
        config(64, 0)
        desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                        dim=0).to(output)
        for k in range(num_samples):
            assert test[:, k].allclose(desired, atol=0.0)
            desired = desired + 1.0
        input = ones_like(test, device=output, requires_grad=True)
        mean(test - input).backward()
    # test gen_num_trajs() computing on G.P.U. and output to the C.P.U.
    if is_available():
        config(64, 0)
        dims_state = (4, 3)
        num_states = 4
        num_samples = 4
        compute = 'cuda'
        output = 'cpu'
        ds = DiscreteSystemExample(dims_state, True)
        def gen_ic() -> Tensor:  # noqa: E306
            return randn(num_states,)
        test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                             compute=compute, output=output, pbar=False)
        assert test.device.type == output
        assert tuple(test.shape) == (num_traj, num_samples) + dims_state
        config(64, 0)
        desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                        dim=0).to(output)
        for k in range(num_samples):
            assert test[:, k].allclose(desired, atol=0.0)
            desired = desired + 1.0
        input = ones_like(test, device=output, requires_grad=True)
        mean(test - input).backward()
    # test gen_num_trajs() computing on C.P.U. and output to the G.P.U.
    if is_available():
        config(64, 0)
        dims_state = (4, 3)
        num_states = 4
        num_samples = 4
        compute = 'cpu'
        output = 'cuda'
        ds = DiscreteSystemExample(dims_state, True)
        def gen_ic() -> Tensor:  # noqa: E306
            return randn(dims_state)
        test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                             compute=compute, output=output, pbar=False)
        assert test.device.type == output
        assert tuple(test.shape) == (num_traj, num_samples) + dims_state
        config(64, 0)
        desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                        dim=0).to(output)
        for k in range(num_samples):
            assert test[:, k].allclose(desired, atol=0.0)
            desired = desired + 1.0
        input = ones_like(test, device=output, requires_grad=True)
        mean(test - input).backward()
    # test gen_num_trajs() computing on G.P.U. and output to the C.P.U. for a
    # dataset larger than the G.P.U.'s available memory
    if is_available():
        with no_grad():
            _, tot_mem = mem_get_info()
            num_traj = 400
            dims_state = (40, 30)
            num_states = prod(dims_state)
            mem_per_sample = num_traj * randn((1,)).element_size() * num_states
            num_samples = int(1.25 * tot_mem / mem_per_sample)
            compute = 'cuda'
            output = 'cpu'
            ds = DiscreteSystemExample(dims_state, True)
            config(64, 0)
            def gen_ic() -> Tensor:  # noqa: E306
                return randn(dims_state)
            test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                                 compute=compute, output=output, pbar=False)
            assert test.device.type == output
            assert tuple(test.shape) == (num_traj, num_samples) + dims_state
            config(64, 0)
            desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                            dim=0).to(output)
            for k in range(num_samples):
                assert test[:, k].allclose(desired, atol=0.0)
                desired = desired + 1.0
