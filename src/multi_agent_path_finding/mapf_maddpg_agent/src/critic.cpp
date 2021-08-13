#include "mapf_maddpg_agent/critic.h"
#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/network.h"
#include "mapf_environment/environment_core.h"
#include <torch/torch.h>


std::vector<float> Critic::serialize_collective(CollectiveObservation coll_obs) const
{
    std::vector<float> full_obs_actions;
    for (auto& obs: coll_obs) {
        // single serialized vec
        std::vector<float> serialized_obs = Environment::serialize_observation(obs);

        // append
        full_obs_actions.insert(full_obs_actions.end(),
            serialized_obs.begin(), serialized_obs.end());
    }

    return full_obs_actions;
}

Critic::Critic(Net* _net, torch::optim::Optimizer* _optim, float _gamma):
    net(_net), optim(_optim), gamma(_gamma) {}

Value Critic::get_value(CollectiveObservation coll_obs) const
{
    torch::Tensor coll_obs_t = torch::tensor(serialize_collective(coll_obs));
    torch::Tensor value_t = net->forward(coll_obs_t);
    Value value = value_t.item<float>();
    return value;
}

std::vector<Value> Critic::get_value(std::vector<CollectiveObservation> coll_obs) const
{
    std::vector<Value> values;
    for (int i=0; i<coll_obs.size(); i++)
        values.push_back(get_value(coll_obs[i]));

    return values;
}

float Critic::train(std::vector<Experience> experiences)
{
    optim->zero_grad();

    int number_of_exp    = experiences.size();
    int number_of_agents = experiences.front().x.size();
    int obs_size         = Environment::serialize_observation(experiences.front().x.front()).size();

    // create tensors
    torch::Tensor x_t = torch::empty({number_of_exp, number_of_agents * obs_size}, torch::kF32);
    torch::Tensor r_t = torch::empty({number_of_exp, 1}, torch::kF32);
    torch::Tensor x__t = torch::empty({number_of_exp, number_of_agents * obs_size}, torch::kF32);
    torch::Tensor d_t = torch::empty({number_of_exp, 1}, torch::kF32);

    for (int i=0; i<experiences.size(); i++) {
        auto exp = experiences[i];
        x_t[i]   = torch::tensor(serialize_collective(exp.x));
        r_t[i]   = exp.reward;
        x__t[i]  = torch::tensor(serialize_collective(exp.x_));
        d_t[i]   = (int)exp.done;
    }

    torch::Tensor V_x  = net->forward(x_t);
    torch::Tensor V_x_ = net->forward(x__t).detach();

    torch::Tensor V_x_expexted = r_t + gamma * d_t * V_x_;

    torch::Tensor loss = torch::mse_loss(V_x, V_x_expexted);
    loss.backward();
    float loss_val = loss.item<float>();
    optim->step();

    return loss_val;
}

