## Exponentially step network weight torwards target
#
#  target = (1-tau) * target + tau * source, tau==1 performs hard update
def soft_update(target, source, tau):
    assert(0. < tau and tau <= 1.)

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


## Disable gradients
#
#  https://github.com/ku2482/soft-actor-critic.pytorch.git
def grad_false(network):
    for param in network.parameters():
        param.requires_grad = False
