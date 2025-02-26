"""Test the dynml.dyn.discrete.numsolve module.

This module tests the dynml.dyn.discrete.numsolve module.
"""

# import built-in python-package code
from random import seed as python_seed
# import external python-package code
from torch import float64, mean, no_grad, ones_like, randn, set_default_dtype
from torch import stack, Tensor
from torch import manual_seed as torch_manual_seed
from torch.cuda import is_available, mem_get_info
from torch.cuda import manual_seed as cuda_manual_seed
# import internal python-package code
from dynml.dyn.discrete.numsolve import gen_num_trajs
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
    # set torch to float64
    set_default_dtype(float64)
    # test that an error is raised when the compute device is not recognized
    try:
        num_states = 4
        num_traj = 4
        num_samples = 4
        ds = DiscreteSystemExample(num_states)
        def gen_ic_1() -> Tensor:  # noqa: E306
            return randn(num_states,)
        gen_num_trajs(ds, gen_ic_1, num_traj, num_samples, compute='unknown')
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == "The compute device is not recognized."
    # test that an error is raised when the output device is not recognized
    try:
        num_states = 4
        num_traj = 4
        num_samples = 4
        ds = DiscreteSystemExample(num_states)
        def gen_ic_2() -> Tensor:  # noqa: E306
            return randn(num_states,)
        gen_num_trajs(ds, gen_ic_2, num_traj, num_samples, output='unknown')
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == "The output device is not recognized."
    # test that an error is raised when the initial condition's number of
    # states does not match the discrete system's number of states
    try:
        num_states = 4
        num_traj = 4
        num_samples = 4
        ds = DiscreteSystemExample(num_states)
        def gen_ic_incorrect() -> Tensor:  # noqa: E306
            return randn(num_states - 1,)
        gen_num_trajs(ds, gen_ic_incorrect, num_traj, num_samples)
        raise NotImplementedError("Test not implemented")
    except ValueError as exc:
        assert str(exc) == ("The initial condition's number of states does "
                            + "not match the discrete system's number of "
                            + "states.")
    # test gen_num_trajs() computing on C.P.U. and output to the C.P.U. and
    # backwards differentiation with respect to gen_num_trajs() output
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    num_states = 4
    num_traj = 4
    num_samples = 4
    compute = 'cpu'
    output = 'cpu'
    ds = DiscreteSystemExample(num_states)
    def gen_ic() -> Tensor:  # noqa: E306
        return randn(num_states,)
    test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                         compute=compute, output=output, pbar=False)
    assert test.device.type == output
    assert tuple(test.shape) == (num_traj, num_samples, num_states)
    python_seed(0)
    torch_manual_seed(0)
    cuda_manual_seed(0)
    desired = stack(tuple(gen_ic() for _ in range(num_traj)), dim=0).to(output)
    for k in range(num_samples):
        assert test[:, k, :].allclose(desired, atol=0.0)
        desired = desired + 1.0
    mean(test - ones_like(test, device=output)).backward()
    # test gen_num_trajs() computing on G.P.U. and output to the G.P.U.
    if is_available():
        python_seed(0)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        num_states = 4
        num_traj = 4
        num_samples = 4
        compute = 'cuda'
        output = 'cuda'
        ds = DiscreteSystemExample(num_states)
        def gen_ic() -> Tensor:  # noqa: E306
            return randn(num_states,)
        test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                             compute=compute, output=output, pbar=False)
        assert test.device.type == output
        assert tuple(test.shape) == (num_traj, num_samples, num_states)
        python_seed(0)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                        dim=0).to(output)
        for k in range(num_samples):
            assert test[:, k, :].allclose(desired, atol=0.0)
            desired = desired + 1.0
        input = ones_like(test, device=output, requires_grad=True)
        mean(test - input).backward()
    # test gen_num_trajs() computing on G.P.U. and output to the C.P.U.
    if is_available():
        python_seed(0)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        num_traj = 4
        num_states = 4
        num_samples = 4
        compute = 'cuda'
        output = 'cpu'
        ds = DiscreteSystemExample(num_states)
        def gen_ic() -> Tensor:  # noqa: E306
            return randn(num_states,)
        test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                             compute=compute, output=output, pbar=False)
        assert test.device.type == output
        assert tuple(test.shape) == (num_traj, num_samples, num_states)
        python_seed(0)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                        dim=0).to(output)
        for k in range(num_samples):
            assert test[:, k, :].allclose(desired, atol=0.0)
            desired = desired + 1.0
        input = ones_like(test, device=output, requires_grad=True)
        mean(test - input).backward()
    # test gen_num_trajs() computing on C.P.U. and output to the G.P.U.
    if is_available():
        python_seed(0)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        _, tot_mem = mem_get_info()
        num_traj = 4
        num_states = 4
        num_samples = 4
        compute = 'cpu'
        output = 'cuda'
        ds = DiscreteSystemExample(num_states)
        def gen_ic() -> Tensor:  # noqa: E306
            return randn(num_states,)
        test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                             compute=compute, output=output, pbar=False)
        assert test.device.type == output
        assert tuple(test.shape) == (num_traj, num_samples, num_states)
        python_seed(0)
        torch_manual_seed(0)
        cuda_manual_seed(0)
        desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                        dim=0).to(output)
        for k in range(num_samples):
            assert test[:, k, :].allclose(desired, atol=0.0)
            desired = desired + 1.0
        input = ones_like(test, device=output, requires_grad=True)
        mean(test - input).backward()
    # test gen_num_trajs() computing on G.P.U. and output to the C.P.U. for a
    # dataset larger than the G.P.U.'s available memory
    if is_available():
        with no_grad():
            _, tot_mem = mem_get_info()
            num_traj = 400
            num_states = 400
            mem_per_sample = num_traj * randn((1,)).element_size() * num_states
            num_samples = int(1.25 * tot_mem / mem_per_sample)
            compute = 'cuda'
            output = 'cpu'
            ds = DiscreteSystemExample(num_states)
            python_seed(0)
            torch_manual_seed(0)
            cuda_manual_seed(0)
            def gen_ic() -> Tensor:  # noqa: E306
                return randn(num_states,)
            test = gen_num_trajs(ds, gen_ic, num_traj, num_samples,
                                 compute=compute, output=output, pbar=False)
            assert test.device.type == output
            assert tuple(test.shape) == (num_traj, num_samples, num_states)
            python_seed(0)
            torch_manual_seed(0)
            cuda_manual_seed(0)
            desired = stack(tuple(gen_ic() for _ in range(num_traj)),
                            dim=0).to(output)
            for k in range(num_samples):
                assert test[:, k, :].allclose(desired, atol=0.0)
                desired = desired + 1.0
