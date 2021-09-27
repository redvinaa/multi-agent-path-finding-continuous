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
    double physics_step_size, laser_max_angle, laser_max_dist, robot_diam;
    int laser_nrays, max_steps, number_of_agents;
    bool draw_laser, draw_noisy_pose;

    // read parameters
    nh.param<std::string>("map_path",     map_path,          "empty_4x4");
    nh.param<int>("number_of_agents",     number_of_agents,  2);
    nh.param<double>("physics_step_size", physics_step_size, 0.1);
    nh.param<double>("laser_max_angle",   laser_max_angle,   45*M_PI/180);
    nh.param<double>("laser_max_dist",    laser_max_dist,    10);
    nh.param<double>("robot_diam",        robot_diam,        0.8);
    nh.param<int>("laser_nrays",          laser_nrays,       10);
    nh.param<int>("max_steps",            max_steps,         150);
    nh.param<bool>("draw_laser",          draw_laser,        true);
    nh.param<bool>("draw_noisy_pose",     draw_noisy_pose,   true);

    std::string full_map_path = ros::package::getPath("mapf_environment") + "/maps/" + map_path + ".jpg";

    env = std::make_shared<Environment>(
        full_map_path,
        number_of_agents,
        physics_step_size,
        1,
        laser_max_angle,
        laser_max_dist,
        robot_diam,
        6,
        2,
        1200,
        laser_nrays,
        max_steps,
        draw_laser,
        draw_noisy_pose);

    // initialize ros communication
    render_publisher     = it.advertise("image", 1);
    physics_timer = nh.createTimer(ros::Duration(physics_step_size), &RosEnvironment::step, this);

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

    std::cout << "====================" << std::endl;
    std::cout << "Sim time: " << env->get_episode_sim_time() << std::endl;

    auto env_step = env->step(coll_action);
    auto rewards  = std::get<1>(env_step);
    std::string info;
    info += "Rewards: ";
    for (auto rew : rewards)
    {
        info += "  " + std::to_string(rew);
    }

    std::cout << info << std::endl;

    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = ros::Time::now();
    cv_img.encoding = "bgr8";
    cv_img.image = env->get_rendered_pic();
    render_publisher.publish(cv_img.toImageMsg());

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
    float eps = 1e-5;

    if (std::abs(action->linear.x) < (1. + eps))
        if (action->linear.x > (0. - eps))
            coll_action[agent_index][0] = action->linear.x;
        else
            ROS_WARN_STREAM("Received illegal linear velocity (below 0)");
    else
    {
        ROS_WARN_STREAM("Received linear velocity greater than 1 m/s ("
            << action->linear.x
            << "), trimming");
        if (action->linear.x > (0. - eps))
        {
            coll_action[agent_index][0] = 1.;
        }
        else
            coll_action[agent_index][0] = -1;
    }

    if (std::abs(action->angular.z) < (M_PI/2 + eps))
        coll_action[agent_index][1] = action->angular.z;
    else
    {
        ROS_WARN_STREAM("Received angular velocity greater than "
            << M_PI/2
            << " rad/s ("
            << action->angular.z
            << "), trimming");
        if (action->angular.z > (0. - eps))
        {
            coll_action[agent_index][1] = M_PI/2.;
        }
        else
            coll_action[agent_index][1] = -M_PI/2;
    }
}
