#include "mapf_maddpg_agent/actor.h"
#include "mapf_maddpg_agent/critic.h"
#include "mapf_maddpg_agent/network.h"
#include "mapf_maddpg_agent/types.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>

int main(int argc, char** argv)
{
    Net net(10);
    torch::optim::SGD optim(net.parameters(), 0.01);
    Actor actor(1, &net, &optim);
}
