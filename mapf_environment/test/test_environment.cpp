// Copyright 2021 Reda Vince

#include "mapf_environment/environment.h"
#include "mapf_environment/pathfinder.h"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
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
            environment =
                new Environment(image_path, std::make_tuple(4., 4.), 1, std::rand(), 30, 0.5);
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

TEST_F(EnvironmentFixture, testReset)
{
    EXPECT_TRUE(environment->done);
    environment->reset();
    EXPECT_FALSE(environment->done);
}

TEST_F(EnvironmentFixture, testContact)
{
    environment->reset();

    auto out = environment->step_physics();
    EXPECT_FALSE(environment->collisions[0]);

    // test collision between agent and wall
    environment->reset();
    float robot_radius = environment->robot_radius;
    environment->agent_bodies[0]->SetTransform(b2Vec2(robot_radius, 4.-robot_radius), 0);
    environment->goal_positions[0].Set(4.-robot_radius, 4.-robot_radius);

    out = environment->step_physics();
    EXPECT_TRUE(environment->collisions[0]);

    // test collision between agents
    environment->reset();
    environment->agent_bodies[0]->SetTransform(b2Vec2(robot_radius+0.1, 4.-robot_radius-0.1), 0);
    environment->goal_positions[0].Set(4.-robot_radius, 4.-robot_radius);
    environment->agent_bodies[1]->SetTransform(b2Vec2(3*robot_radius+0.2, 4.-robot_radius-0.1), M_PI);
    environment->goal_positions[1].Set(4.-robot_radius, 4.-robot_radius);
    // Now there 2 agents, both in the upper row, facing each other,
    // and there is 0.1 meters distance between them
    // with a total of 2 m/s, and dt=0.01, it should take
    // 0.1/(2*0.01)=5 time steps for them to collide.
    // Actually after 5 steps, they almost collide, so it takes a 6th step for them to do so

    std::vector<float> action(2);
    action[0] = 1;
    action[1] = 0;
    environment->process_action(0, action);
    environment->process_action(1, action);

    for (int i=0; i < 6; i++)
    {
        environment->step_physics();
        // auto find_coll = std::find(environment->collisions.begin(), environment->collisions.end(), true);
        // EXPECT_TRUE(find_coll == environment->collisions.end());  // all false
    }

    environment->step_physics();

    // auto find_coll = std::find(environment->collisions.begin(), environment->collisions.end(), false);
    // EXPECT_TRUE(find_coll == environment->collisions.end());  // all true
}

TEST_F(EnvironmentFixture, testMovement)
{
    environment->reset();

    std::vector<float> action(2);
    action[0] = 1;
    action[1] = -0.5;
    environment->process_action(0, action);

    environment->step_physics();
    b2Vec2 lin_vel = environment->agent_bodies[0]->GetLinearVelocity();
    float ang_vel = environment->agent_bodies[0]->GetAngularVelocity();
    EXPECT_EQ(lin_vel.Length(), 1.);
    EXPECT_EQ(ang_vel, -0.5);
}

TEST_F(EnvironmentFixture, testObservation)
{
    EXPECT_EQ(environment->collision_reward, -0.5);
    EXPECT_EQ(environment->goal_reaching_reward, 1.);
    EXPECT_EQ(environment->max_steps, 30);

    environment->reset();

    std::vector<float> action(2);
    action[0] = 1;
    action[1] = -0.5;
    std::vector<std::vector<float>> actions = {action, action};

    auto env_step = environment->step(actions);

    b2Vec2 agent_pose(std::get<0>(env_step)[0][0], std::get<0>(env_step)[0][1]);
    // b2Transform agent_tf_expected = environment->agent_bodies[0]->GetTransform();
    // EXPECT_EQ(agent_pose, agent_tf_expected.p);  // not true because of noise

    // not true because of noise
    // EXPECT_EQ(env_obs.observations[0].agent_twist.x, 1);
    // EXPECT_EQ(env_obs.observations[0].agent_twist.y, 0);
    // EXPECT_EQ(env_obs.observations[0].agent_twist.z, -0.5);
    EXPECT_EQ(std::get<2>(env_step)[0], false);

    // test collision reward
    float robot_radius = environment->robot_radius;
    environment->agent_bodies[0]->SetTransform(b2Vec2(robot_radius, 4.-robot_radius), 0);
    environment->goal_positions[0].Set(4.-robot_radius, 4.-robot_radius);

    auto obs_and_rewards = environment->step_physics();
    EXPECT_TRUE(environment->collisions[0]);

    // EXPECT_EQ(std::get<1>(obs_and_rewards)[0], environment->collision_reward);

    // test goal reaching reward
    environment->agent_bodies[0]->SetTransform(environment->goal_positions[0], 0);
    obs_and_rewards = environment->step_physics();
    // EXPECT_EQ(environment->is_done(), false);
    // EXPECT_EQ(std::get<1>(obs_and_rewards)[0], 1.);
    // TODO(redvinaa): Include goal_distance
}

TEST_F(EnvironmentFixture, testRender)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> dist(0., 1.);

    auto obs = environment->reset();

    std::vector<std::vector<float>> actions(1);

    while (not environment->is_done())
    {
        environment->render(0, true);

        for (auto& a : actions)
        {
            a.resize(2);
            a[0] = dist(generator);
            a[1] = (dist(generator) - 0.5) * M_PI;
        }

        environment->step(actions);
    }
}

TEST_F(EnvironmentFixture, testOsSpace)
{
    auto shape = environment->get_observation_space();
    auto obs = environment->reset();

    EXPECT_EQ(shape.size(), obs.size());
    EXPECT_EQ(shape[0], obs[0].size());
}

TEST(AStar, testAStar)
{
    cv::Mat img = cv::Mat::zeros(cv::Size(10, 10), CV_8UC1);
    img = 255;

    // add an obstacle
    cv::circle(img, cv::Point(5, 5), 2, 0, -1);

    AStar astar(img, true, true);

    auto generate_pos = [img]()
    {
        int start_x, start_y, goal_x, goal_y;

        while (true)
        {
            start_x = std::rand() % 10;
            start_y = std::rand() % 10;
            if (static_cast<int>(img.at<unsigned char>(start_x, start_y)) > 255/2)
                break;
        }

        while (true)
        {
            goal_x = std::rand() % 10;
            goal_y = std::rand() % 10;
            if ((static_cast<int>(img.at<unsigned char>(goal_x, goal_y)) > 255/2)
                and not ((start_x == goal_x) and (start_y == goal_y)))
                break;
        }

        return std::make_tuple(start_x, start_y, goal_x, goal_y);
    };


    for (int i=0; i < 20; i++)
    {
        cv::Mat img_cpy;
        img.copyTo(img_cpy);

        int start_x, start_y, goal_x, goal_y;
        std::tie(start_x, start_y, goal_x, goal_y) = generate_pos();

        t_path route;
        float dist;
        std::tie(route, dist) = astar.find(start_x, start_y, goal_x, goal_y);
        std::cout << "Distance: " << dist << std::endl;

        cv::resize(img_cpy, img_cpy, cv::Size(700, 700), 0, 0, cv::INTER_AREA);

        // draw start and end
        cv::Point center;
        center.x = (start_x+0.5) * 700 / 10;
        center.y = (start_y+0.5) * 700 / 10;
        cv::circle(img_cpy, center, 700/10/2, 0, -1); // start is full
        center.x = (goal_x+0.5) * 700 / 10;
        center.y = (goal_y+0.5) * 700 / 10;
        cv::circle(img_cpy, center, 700/10/2, 0);

        // draw path
        int n_pts = route.size() + 1;
        cv::Point points[n_pts];
        points[0] = cv::Point((start_x+0.5) * 700 / 10, (start_y+0.5) * 700 / 10);
        for (int i=1; i < n_pts; i++)
        {
            int pt_x = (std::get<0>(route[i-1])+0.5) * 700 / 10;
            int pt_y = (std::get<1>(route[i-1])+0.5) * 700 / 10;
            points[i] = cv::Point(pt_x, pt_y);
        }
        const cv::Point* ps = points;
        cv::polylines(img_cpy, &ps, &n_pts, 1, /*closed=*/false, 255/2, 2);

        cv::imshow("Planner", img_cpy);
        cv::waitKey();
    }
}

int main(int argc, char **argv)
{
    std::srand(time(0));
    testing::InitGoogleTest(&argc, argv);
    testing::GTEST_FLAG(filter) = "EnvironmentFixture.testRender";
    return RUN_ALL_TESTS();
}
