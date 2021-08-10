#include "mapf_maddpg_agent/actor.h"
#include "mapf_maddpg_agent/critic.h"
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>

int main(int argc, char** argv) {
	torch::Tensor t = torch::rand({2, 3});
	std::cout << t << std::endl;
}
