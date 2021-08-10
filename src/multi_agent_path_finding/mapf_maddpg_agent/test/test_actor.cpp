#include <gtest/gtest.h>
#include <torch/torch.h>
#include "mapf_maddpg_agent/actor.h"

class ActorFixture : public testing::Test {
	protected:
		Actor* actor;

		void SetUp() override {
			actor = new Actor();
		}

		void TearDown() override {
			delete actor;
		}
};

TEST_F(ActorFixture, testConstructor) {
}

int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
