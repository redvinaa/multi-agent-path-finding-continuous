// Copyright 2021 Reda Vince

#include "mapf_environment/types.h"
#include "mapf_maddpg_agent/types.h"
#include "mapf_maddpg_agent/network.h"
#include "mapf_maddpg_agent/critic.h"
#include "mapf_maddpg_agent/actor.h"
#include "mapf_environment/environment.h"
#include <gtest/gtest.h>
#include <torch/torch.h>
#include <ros/package.h>
#include <stdlib.h>
#include <vector>
#include <string>

TEST(NetworkTest, testLossDecreases)
{
    // Test if parameters are changed during training.
    // Networks with different numbers of hidden layers are tested.

    Net net(10, 2, 10, 3);

    auto params_before = net.parameters();
    for (auto& param_group : params_before)
        param_group = param_group.clone();

    torch::Tensor dummy_input  = torch::rand({5, 10}, torch::kF32) + 4;
    EXPECT_EQ(dummy_input.dtype(), torch::kF32);

    torch::Tensor dummy_output = torch::rand({5, 2}, torch::kF32) + 6;
    EXPECT_EQ(dummy_output.dtype(), torch::kF32);

    torch::optim::SGD optimizer(net.parameters(), /* lr */0.001);
    std::vector<float> losses;

    for (int epoch=0; epoch < 5; epoch++)
    {
        optimizer.zero_grad();

        torch::Tensor prediction = net.forward(dummy_input);
        EXPECT_EQ(prediction.dtype(), torch::kF32);
        EXPECT_EQ(prediction.sizes(), dummy_output.sizes());
        EXPECT_FALSE(prediction.equal(torch::zeros({5, 2})));

        torch::Tensor loss = torch::mse_loss(prediction, dummy_output);
        losses.push_back(loss.item<float>());

        loss.backward();
        optimizer.step();
    }

    for (int i=0; i < net.parameters().size(); i++)
        EXPECT_FALSE(params_before[i].equal(net.parameters()[i]));

    EXPECT_TRUE(losses.front() > losses.back()) << "Losses have not gone down";
}


class CriticFixture : public testing::Test
{
    protected:
        Critic* critic;
        Net* net;
        torch::optim::Optimizer* optim;
        Environment* environment;
        int number_of_agents;
        unsigned int seed = 0;

        Action get_random_action()
        {
            Action action;
            action.linear.x = rand_r(&seed);
            action.angular.z = rand_r(&seed);

            return action;
        }

        void SetUp() override
        {
            number_of_agents = 2;

            std::string pkg_path = ros::package::getPath("mapf_environment");
            std::string image_path = pkg_path + "/maps/test_4x4.jpg";
            environment = new Environment(image_path);
            environment->reset();

            for (int i=0; i < number_of_agents; i++)
                environment->add_agent();

            int input_size = number_of_agents * environment->get_observation_size();

            net = new Net(input_size, 1, 10, 1);
            optim = new torch::optim::Adam(net->parameters(), /*lr=*/1e-3);
            critic = new Critic(net, optim, 0.9);
        }

        void TearDown() override
        {
            delete net;
            delete optim;
            delete critic;
            delete environment;
        }
};

TEST_F(CriticFixture, testGetValues)
{
    CollectiveObservation coll_obs;
    for (int i=0; i < number_of_agents; i++)
        coll_obs.push_back(environment->get_observation(0));

    float value = critic->get_value(coll_obs);
    EXPECT_TRUE(value != 0.);

    std::vector<CollectiveObservation> coll_obs_v;
    coll_obs_v.push_back(coll_obs);
    coll_obs_v.push_back(coll_obs);

    auto values = critic->get_value(coll_obs_v);
}

TEST_F(CriticFixture, testTraining)
{
    // collect experiences
    std::vector<Experience> experiences;

    Experience exp;
    Observation obs_0, obs_1;
    Action action;

    obs_0 = environment->get_observation(0);
    obs_1 = environment->get_observation(1);
    exp.x_.push_back(obs_0);
    exp.x_.push_back(obs_1);

    for (int step=0; step < 10; step++)
    {
        exp.x = exp.x_;

        exp.a.clear();
        action = get_random_action();
        environment->process_action(0, action);
        exp.a.push_back(action);
        action = get_random_action();
        environment->process_action(1, action);
        exp.a.push_back(action);

        environment->step_physics();

        obs_0 = environment->get_observation(0);
        obs_1 = environment->get_observation(1);

        exp.x_.clear();
        exp.x_.push_back(obs_0);
        exp.x_.push_back(obs_1);

        EXPECT_TRUE(exp.x != exp.x_);  // check if elements were deep copied

        exp.reward = (obs_0.reward + obs_1.reward) / 2.;
        exp.done = environment->is_done();

        experiences.push_back(exp);

        if (environment->is_done())
            break;
    }

    // train
    float loss = critic->train(experiences);
    EXPECT_TRUE(loss != 0.);
}


class ActorFixture : public testing::Test
{
    protected:
        Actor* actor;
        Net* net;
        torch::optim::Optimizer* optim;

        void SetUp() override
        {
            net = new Net(10, 2, 10, 1);
            optim = new torch::optim::Adam(net->parameters(), 1e-3);
            actor = new Actor(net, optim, 1);  // TODO(redvinaa) Figure out entropy
        }

        void TearDown() override
        {
            delete net;
            delete optim;
            delete actor;
        }
};

TEST_F(ActorFixture, testConstructor) {}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
