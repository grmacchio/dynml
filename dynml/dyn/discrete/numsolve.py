"""Define a generation method for numerical trajectories.

This module defines a generation method for numerical trajectories. A numerical
trajectory is an approximation of a discrete dynamical system trajectory using
floating-point arithmetic.
"""


# import built-in python-package code
from tqdm import tqdm  # type: ignore  # no stubs
from typing import Callable
# import external python-package code
from torch import Tensor, zeros
from torch.cuda import max_memory_allocated, mem_get_info
# import internal python-package code
from dynml.dyn.discrete.system import DiscreteSystem


# export public code
__all__ = ["gen_num_trajs"]


# define a generation method for numerical trajectories
def gen_num_trajs(discrete_sys: DiscreteSystem,  # noqa: C901
                  gen_ic: Callable[[], Tensor], num_traj: int,
                  num_samples: int, compute: str = 'cpu', output: str = 'cpu',
                  pbar: bool = True) -> Tensor:
    """Generate any number of numerical trajectories.

    This method generates any number of numerical trajectories for a given
    discrete dynamical system. Furthermore, this method is G.P.U.-accelerated:
    once :math:`90\\%` of available G.P.U. memory is determined, the numerical
    trajectories are generated in temporal batches to avoid G.P.U. memory
    overflow. The device on which the numerical trajectories are computed and
    stored is determined by the ``compute`` and ``output`` device strings,
    respectively.

    | **Args**
    |   ``discrete_sys`` (``DiscreteSystem``): the discrete dynamical system
    |   ``gen_ic`` (``Callable[[], Tensor]``): the initial condition method,
            which returns an initial condition with shape
            ``(...,) + (discrete_sys.num_states,)``
    |   ``num_traj`` (``int``): the number of trajectories
    |   ``num_samples`` (``int``): the number of samples
    |   ``compute`` (``str``): the compute device string
    |   ``output`` (``str``): the output device string
    |   ``pbar`` (``bool``): the progress bar boolean

    | **Return**
    |   ``Tensor``: the numerical trajectories with shape
            ``(self.num_traj, self.num_samples, discrete_sys.num_states)``

    | **Raises**
    |   ``ValueError``: if the initial condition's number of states does not
            match the discrete system's number of states

    | **References**
    |   None
    """
    # check if the compute and output devices are recognized
    if compute not in ['cpu', 'cuda'] and 'cuda' not in compute:
        raise ValueError("The compute device is not recognized.")
    if output not in ['cpu', 'cuda'] and 'cuda' not in output:
        raise ValueError("The output device is not recognized.")
    # check if the discrete system's number of states matches the initial
    # condition's number of states
    first_ic = gen_ic()
    num_states = first_ic.shape[0]
    if num_states != discrete_sys.num_states:
        raise ValueError("The initial condition's number of states does "
                         "not match the discrete system's number of states.")
    # initialize the numerical trajectories storage tensor
    traj = zeros((num_traj, num_samples, num_states), dtype=first_ic.dtype,
                 device=output)
    # set the initial conditions
    current = zeros((num_traj, num_states), dtype=first_ic.dtype,
                    device=compute)
    current[0, :] = first_ic.to(compute)
    for i in range(num_traj - 1):
        current[i + 1, :] = gen_ic().to(compute)
    traj[:, 0, :] = current.to(output)
    # determine the original device of the discrete system
    orig_device = next(discrete_sys.parameters()).device
    # send the discrete system to the compute device
    discrete_sys = discrete_sys.to(compute)
    # determine the batch size of trajectories
    pbar_steps = tqdm(total=num_samples - 1, disable=not pbar)
    if compute == 'cpu':
        current = discrete_sys.map(current)
        pbar_steps.update(1)
        traj[:, 1, :] = current.to(output)
        batch_size = num_samples - 1
    elif 'cuda' in compute:
        mem_baseline = max_memory_allocated()
        current = discrete_sys.map(current)
        pbar_steps.update(1)
        traj[:, 1, :] = current.to(output)
        mem_spike = max_memory_allocated() - mem_baseline
        avail_mem, _ = mem_get_info()
        mem_per_sample = num_traj * 1 * num_states * traj.element_size()
        batch_size = int((0.9 * avail_mem - mem_spike) / mem_per_sample)
    # map the initial conditions
    for sample_idx in range(1, num_samples, batch_size):
        subbatch_size = min(batch_size, num_samples - sample_idx)
        if sample_idx != 1:
            del batch
        batch: Tensor = zeros((num_traj, subbatch_size, num_states),
                              dtype=first_ic.dtype, device=compute)
        if sample_idx == 1:
            batch[:, 0, :] = current
        else:
            current = discrete_sys.map(current)
            pbar_steps.update(1)
            batch[:, 0, :] = current
        for i in range(subbatch_size - 1):
            current = discrete_sys.map(current)
            pbar_steps.update(1)
            batch[:, i + 1, :] = current
        traj[:, sample_idx:sample_idx + subbatch_size] = batch.to(output)
    # send the discrete system back to its original device
    discrete_sys = discrete_sys.to(orig_device)
    # return the numerical trajectories
    return traj
