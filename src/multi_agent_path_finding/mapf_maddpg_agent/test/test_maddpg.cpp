#include <gtest/gtest.h>
#include <torch/torch.h>
#include "mapf_maddpg_agent/actor.h"
#include "mapf_maddpg_agent/network.h"

class ActorFixture : public testing::Test
{
    protected:
        Actor* actor;
        Net* net;
        torch::optim::Optimizer* optim;

        void SetUp() override {
            net = new Net(10, 2, 10, 1);
            optim = new torch::optim::Adam(net->parameters(), 1e-3);
            actor = new Actor(0, net, optim);
        }

        void TearDown() override {
            delete net;
            delete optim;
            delete actor;
        }
};

TEST_F(ActorFixture, testConstructor)
{
}

TEST(NetworkTest, testLossDecreases)
{
    // Test if parameters are changed during training.
    // Networks with different numbers of hidden layers are tested.

    Net net(10, 2, 10, 3);

    auto params_before = net.parameters();
    for (auto& param_group: params_before)
        param_group = param_group.clone();

    torch::Tensor dummy_input  = torch::rand({5, 10}, torch::kF32) + 4;
    EXPECT_EQ(dummy_input.dtype(), torch::kF32);

    torch::Tensor dummy_output = torch::rand({5, 2}, torch::kF32) + 6;
    EXPECT_EQ(dummy_output.dtype(), torch::kF32);

    torch::optim::SGD optimizer(net.parameters(), /* lr */0.001);
    std::vector<float> losses;

    for (int epoch=0; epoch<5; epoch++) {

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

    for (int i=0; i<net.parameters().size(); i++)
        EXPECT_FALSE(params_before[i].equal(net.parameters()[i]));

    EXPECT_TRUE(losses.front() > losses.back()) << "Losses have not gone down";
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
