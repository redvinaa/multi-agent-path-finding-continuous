// Copyright 2021 Reda Vince

#include "mapf_environment/environment.h"
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <gtest/gtest.h>
#include <geometry_msgs/Twist.h>

class EnvironmentFixture : public testing::Test
{
    protected:
        Environment* environment;

        void SetUp() override
        {
            std::string pkg_path = ros::package::getPath("mapf_environment");
            std::string image_path = pkg_path + "/maps/test_4x4.jpg";
            environment = new Environment(image_path, /*physics_step_size=*/0.01, /*step_multiply/*/1);
        }

        void TearDown() override
        {
            delete environment;
        }
};

TEST_F(EnvironmentFixture, testConstructor)
{
    EXPECT_FALSE(environment->map_image_raw.empty());
    EXPECT_NE(environment->robot_radius, 0);
    EXPECT_TRUE(environment->map_image.size().height == environment->render_height);
}

TEST_F(EnvironmentFixture, testMap)
{
    EXPECT_EQ(environment->map_image_raw.size(), cv::Size(4, 4));
    EXPECT_EQ(environment->map_image_raw.channels(), 1);
    EXPECT_TRUE(environment->map_image_raw.at<uchar>(0, 0) > 125);
    EXPECT_TRUE(environment->map_image_raw.at<uchar>(1, 0) < 125);
    EXPECT_EQ(environment->map_image.channels(), 3);
    EXPECT_EQ(environment->map_image.size().height, environment->render_height);

    // cv::Mat map = environment->render();
    // cv::imshow("test_4x4.jpg", map);
    // cv::waitKey(0);
}

TEST_F(EnvironmentFixture, testPhysics)
{
    bool hit;

    // test that nothing is in the upper left cell
    b2Transform transform;
    b2Vec2 point;

    point.Set(0.5f, 3.5f);
    hit = false;
    for (auto it=environment->obstacle_bodies.begin(); it != environment->obstacle_bodies.end(); it++)
    {
        auto body = *it;
        transform = body->GetTransform();

        auto shape = body->GetFixtureList()->GetShape();
        ASSERT_TRUE(shape != NULL);

        hit = shape->TestPoint(transform, point);
        if (hit == true)
            break;
    }
    EXPECT_FALSE(hit);

    // test that there is an obstacle right below that

    point.Set(0.5f, 2.5f);
    hit = false;
    for (auto it=environment->obstacle_bodies.begin(); it != environment->obstacle_bodies.end(); it++)
    {
        b2Body* body = *it;
        transform = body->GetTransform();

        b2Fixture* fixture = body->GetFixtureList();
        b2Shape* shape = fixture->GetShape();
        ASSERT_TRUE(shape != NULL);

        hit = shape->TestPoint(transform, point);
        if (hit == true)
            break;
    }
    EXPECT_TRUE(hit);
}

TEST_F(EnvironmentFixture, testAddAndRemoveAgent)
{
    EXPECT_EQ(environment->number_of_agents, 0);
    int idx = environment->add_agent();
    EXPECT_EQ(idx, 0);
    EXPECT_EQ(environment->number_of_agents, 1);

    EXPECT_TRUE(environment->agent_bodies[0]->GetLinearVelocity() == b2Vec2(0, 0));
    EXPECT_TRUE(environment->agent_bodies[0]->GetAngularVelocity() == 0.);

    EXPECT_EQ(1, environment->agent_bodies.size());
    EXPECT_EQ(1, environment->goal_positions.size());
    EXPECT_EQ(1, environment->agent_colors.size());
    EXPECT_EQ(1, environment->agent_lin_vel.size());
    EXPECT_EQ(1, environment->agent_ang_vel.size());
    EXPECT_EQ(1, environment->collisions.size());
    EXPECT_EQ(1, environment->laser_scans.size());

    EXPECT_EQ(environment->laser_nrays, environment->laser_scans[0].size());

    environment->remove_agent(0);
    EXPECT_EQ(environment->number_of_agents, 0);

    EXPECT_EQ(0, environment->agent_bodies.size());
    EXPECT_EQ(0, environment->goal_positions.size());
    EXPECT_EQ(0, environment->agent_colors.size());
    EXPECT_EQ(0, environment->agent_lin_vel.size());
    EXPECT_EQ(0, environment->agent_ang_vel.size());
    EXPECT_EQ(0, environment->collisions.size());
    EXPECT_EQ(0, environment->laser_scans.size());
}

TEST_F(EnvironmentFixture, testReset)
{
    EXPECT_TRUE(environment->done);
    environment->reset();
    EXPECT_FALSE(environment->done);
}

TEST_F(EnvironmentFixture, testContact)
{
    environment->reset();
    environment->add_agent();

    environment->step_physics();
    EXPECT_FALSE(environment->collisions[0]);

    // test collision between agent and wall
    environment->reset();
    float robot_radius = environment->robot_radius;
    environment->agent_bodies[0]->SetTransform(b2Vec2(robot_radius, 4.-robot_radius), 0);
    environment->goal_positions[0].Set(4.-robot_radius, 4.-robot_radius);

    environment->step_physics();
    EXPECT_TRUE(environment->collisions[0]);

    // test collision between agents
    environment->reset();
    environment->add_agent();
    environment->agent_bodies[0]->SetTransform(b2Vec2(robot_radius+0.1, 4.-robot_radius-0.1), 0);
    environment->goal_positions[0].Set(4.-robot_radius, 4.-robot_radius);
    environment->agent_bodies[1]->SetTransform(b2Vec2(3*robot_radius+0.2, 4.-robot_radius-0.1), M_PI);
    environment->goal_positions[1].Set(4.-robot_radius, 4.-robot_radius);
    // Now there 2 agents, both in the upper row, facing each other,
    // and there is 0.1 meters distance between them
    // with a total of 2 m/s, and dt=0.01, it should take
    // 0.1/(2*0.01)=5 time steps for them to collide.
    // Actually after 5 steps, they almost collide, so it takes a 6th step for them to do so

    Action action;
    action.x = 1;
    action.z = 0;
    environment->process_action(0, action);
    environment->process_action(1, action);

    for (int i=0; i < 6; i++)
    {
        environment->step_physics();
        auto find_coll = std::find(environment->collisions.begin(), environment->collisions.end(), true);
        EXPECT_TRUE(find_coll == environment->collisions.end());  // all false
    }

    environment->step_physics();

    auto find_coll = std::find(environment->collisions.begin(), environment->collisions.end(), false);
    EXPECT_TRUE(find_coll == environment->collisions.end());  // all true
}

TEST_F(EnvironmentFixture, testMovement)
{
    environment->reset();
    environment->add_agent();

    Action action;
    action.x = 1;
    action.z = -0.5;
    environment->process_action(0, action);

    environment->step_physics();
    b2Vec2 lin_vel = environment->agent_bodies[0]->GetLinearVelocity();
    float ang_vel = environment->agent_bodies[0]->GetAngularVelocity();
    EXPECT_EQ(lin_vel.Length(), 1.);
    EXPECT_EQ(ang_vel, -0.5);
}

TEST_F(EnvironmentFixture, testObservation)
{
    EXPECT_EQ(environment->step_reward, -1);
    EXPECT_EQ(environment->collision_reward, -1);
    EXPECT_EQ(environment->goal_reaching_reward, 0);

    environment->reset();
    environment->add_agent();

    Action action;
    action.x = 1;
    action.z = -0.5;
    CollectiveAction actions = {action};

    EnvStep env_obs = environment->step(actions);
    environment->render(2000);

    b2Transform agent_tf_expected = environment->agent_bodies[0]->GetTransform();
    b2Vec2 agent_pose(env_obs.observations[0].agent_pose.x, env_obs.observations[0].agent_pose.y);
    EXPECT_EQ(agent_pose, agent_tf_expected.p);

    EXPECT_EQ(env_obs.observations[0].agent_twist.x, 1);
    EXPECT_EQ(env_obs.observations[0].agent_twist.y, 0);
    EXPECT_EQ(env_obs.observations[0].agent_twist.z, -0.5);
    EXPECT_EQ(env_obs.observations[0].reward, -1);
    EXPECT_EQ(env_obs.done, false);

    // test collision reward
    float robot_radius = environment->robot_radius;
    environment->agent_bodies[0]->SetTransform(b2Vec2(robot_radius, 4.-robot_radius), 0);
    environment->goal_positions[0].Set(4.-robot_radius, 4.-robot_radius);

    environment->step_physics();

    EXPECT_TRUE(environment->collisions[0]);
    env_obs.observations[0] = environment->get_observation(0);
    env_obs.done = environment->done;
    EXPECT_EQ(env_obs.done, false);
    EXPECT_EQ(env_obs.observations[0].reward, -2);

    // test goal reaching reward
    environment->agent_bodies[0]->SetTransform(environment->goal_positions[0], 0);
    environment->step_physics();
    env_obs.observations[0] = environment->get_observation(0);
    env_obs.done = environment->done;
    EXPECT_EQ(env_obs.done, true);
    EXPECT_EQ(env_obs.observations[0].reward, 0);
}

TEST_F(EnvironmentFixture, testSerialize)
{
    environment->reset();
    environment->add_agent();
    auto obs = environment->get_observation(0);
    obs.reward = 0;
    auto ser_obs = Environment::serialize_observation(obs);
    EXPECT_TRUE(obs.agent_pose.x == Environment::deserialize_observation(ser_obs).agent_pose.x);
    EXPECT_TRUE(obs.scan[0] == Environment::deserialize_observation(ser_obs).scan[0]);
}


int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
