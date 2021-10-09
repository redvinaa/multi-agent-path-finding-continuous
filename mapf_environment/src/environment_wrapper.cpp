// Copyright 2021 Reda Vince

// ros headers
#include <ros/ros.h>
#include <ros/package.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Float32.h>
#include <std_msgs/MultiArrayLayout.h>
#include <std_msgs/MultiArrayDimension.h>
#include <std_msgs/Float32MultiArray.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/LaserScan.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// other headers
#include <mapf_environment/environment_wrapper.h>
#include <mapf_environment/environment.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <stdlib.h>
#include <algorithm>
#include <cmath>
#include <iostream>


RosEnvironment::RosEnvironment(ros::NodeHandle _nh):
        nh(_nh), it(_nh)
{
    double robot_diam;
    int number_of_agents;

    // read parameters
    nh.param<std::string>("map_path",     map_path,          "test_4x4");
    nh.param<int>("number_of_agents",     number_of_agents,  2);
    nh.param<double>("robot_diam",        robot_diam,        0.7);

    std::string full_map_path = ros::package::getPath("mapf_environment") + "/maps/" + map_path + ".jpg";

    env = std::make_shared<Environment>(
        full_map_path,
        std::make_tuple(4, 4),
        number_of_agents,
        0,  // seed
        99999,  // max_steps
        robot_diam,
        0.,  // noise
        0.05,  // physics_step_size
        1);  // step_multiply

    // initialize ros communication
    render_publisher     = it.advertise("image", 1);
    physics_timer = nh.createTimer(ros::Duration(0.05), &RosEnvironment::step, this);

    coll_action.resize(env->get_number_of_agents());
    for (auto& act : coll_action)
        act.resize(2);

    for (int agent_index=0; agent_index < number_of_agents; agent_index++)
    {
        ros::Publisher observation_publisher = nh.advertise<std_msgs::Float32MultiArray>(
            "agent_"+std::to_string(agent_index)+"/observation", 1);
        observation_publishers.push_back(observation_publisher);

        ros::Subscriber action_subscriber = nh.subscribe<geometry_msgs::Twist>(
            "agent_"+std::to_string(agent_index)+"/cmd_vel", 1,
            boost::bind(&RosEnvironment::process_action, this, agent_index, _1));
        action_subscribers.push_back(action_subscriber);
    }

    ROS_INFO("Initialized environment");
}

void RosEnvironment::step(const ros::TimerEvent&)
{
    if (env->is_done())
    {
        ROS_INFO("Episode is over, resetting...");
        env->reset();
    }

    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = ros::Time::now();
    cv_img.encoding = "bgr8";
    cv_img.image = env->get_rendered_pic(true);
    render_publisher.publish(cv_img.toImageMsg());

    std::cout << "====================" << std::endl;
    std::cout << "Sim time:\n  " << env->get_episode_sim_time() << std::endl;

    auto env_step = env->step(coll_action);
    auto rewards  = std::get<1>(env_step);
    std::string info;
    info += "Rewards:\n";
    for (auto rew : rewards)
    {
        info += "  " + std::to_string(rew);
    }

    info += "\nObservations\n";
    for (int obs_index=0; obs_index < env->get_observation_space()[0]; obs_index++)
    {
        for (int agent_index=0; agent_index < env->get_number_of_agents(); agent_index++)
        {
            info += "  " + std::to_string(std::get<0>(env_step)[agent_index][obs_index]);
        }
        info += "\n";
    }

    std::cout << info << std::endl;

    auto observations = std::get<0>(env_step);
    for (int agent_index=0; agent_index < env->get_number_of_agents(); agent_index++)
    {
        std_msgs::Float32MultiArray ros_obs;
        ros_obs.data.clear();
        ros_obs.data = observations[agent_index];
        observation_publishers[agent_index].publish(ros_obs);
    }
}

void RosEnvironment::process_action(int agent_index, const geometry_msgs::TwistConstPtr& action)
{
    coll_action[agent_index][0] = action->linear.x;
    coll_action[agent_index][1] = action->angular.z;
}
