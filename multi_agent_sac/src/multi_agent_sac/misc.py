import numpy as np


## Exponentially step network weight torwards target
#
#  target = (1-tau) * target + tau * source, tau==1 performs hard update
def soft_update(target, source, tau):
    assert(0. < tau and tau <= 1.)

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    soft_update(target, source, 1.)


## Disable gradients
def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False


## Perform gradient descent
def update_params(optim, loss):
    optim.zero_grad()
    loss.backward()
    optim.step()


## Convert normalized actions to environment scale
def from_unit_actions(act: np.ndarray, min_linear_speed: float, max_linear_speed: float, max_angular_speed: float):
    dim = act.ndim

    act = np.copy(act)

    lin_mean  = (min_linear_speed + max_linear_speed) / 2.
    lin_scale = (max_linear_speed - min_linear_speed) / 2.

    if dim == 2:  # act.shape=(n_agents, act_size)
        act[:, 0] = (act[:, 0] * lin_scale) + lin_mean
        act[:, 1] *= max_angular_speed  # angular

    elif dim == 3:  # act.shape=(n_sample, n_agents, act_size)
        act[:, :, 0] = (act[:, :, 0] * lin_scale) + lin_mean
        act[:, :, 1] *= max_angular_speed  # angular

    else:
        raise NotImplementedError

    return act

## Convert environment scale actions to normalized ones
def to_unit_actions(act: np.ndarray, min_linear_speed: float, max_linear_speed: float, max_angular_speed: float):
    dim = act.ndim

    act = np.copy(act)

    lin_mean  = (min_linear_speed + max_linear_speed) / 2.
    lin_scale = (max_linear_speed - min_linear_speed) / 2.

    if dim == 2:  # act.shape=(n_agents, act_size)
        act[:, 0] = (act[:, 0] - lin_mean) / lin_scale
        act[:, 1] /= max_angular_speed  # angular

    elif dim == 3:  # act.shape=(n_sample, n_agents, act_size)
        act[:, :, 0] = (act[:, :, 0] - lin_mean) / lin_scale
        act[:, :, 1] /= max_angular_speed  # angular

    else:
        raise NotImplementedError

    return act
