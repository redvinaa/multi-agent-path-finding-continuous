// Copyright 2021 Reda Vince

// ros headers
#include <ros/ros.h>
#include <ros/package.h>
#include <std_srvs/Empty.h>
#include <std_msgs/Float32.h>
#include <geometry_msgs/Twist.h>
#include <mapf_environment/Observation.h>
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
    std::string default_map_path = ros::package::getPath("mapf_environment") + "/maps/test_4x4.jpg";
    double physics_step_size, laser_max_angle, laser_max_dist, robot_diam;
    int laser_nrays, max_steps, number_of_agents;
    bool draw_laser, draw_noisy_pose;

    // read parameters
    nh.param<std::string>("map_path",     map_path,          default_map_path);
    nh.param<int>("number_of_agents",     number_of_agents,  2);
    nh.param<double>("physics_step_size", physics_step_size, 0.2);
    nh.param<double>("laser_max_angle",   laser_max_angle,   45*M_PI/180);
    nh.param<double>("laser_max_dist",    laser_max_dist,    10);
    nh.param<double>("robot_diam",        robot_diam,        0.8);
    nh.param<int>("laser_nrays",          laser_nrays,       10);
    nh.param<int>("max_steps",            max_steps,         150);
    nh.param<bool>("draw_laser",          draw_laser,        true);
    nh.param<bool>("draw_noisy_pose",     draw_noisy_pose,   true);

    env = std::make_shared<Environment>(
        map_path,
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
    for (int agent_index=0; agent_index < number_of_agents; agent_index++)
    {
        ros::Publisher observation_publisher = nh.advertise<mapf_environment::Observation>(
            "agent_"+std::to_string(agent_index)+"/observation", 1);
        observation_publishers.push_back(observation_publisher);

        ros::Subscriber action_subscriber = nh.subscribe<geometry_msgs::Twist>(
            "agent_"+std::to_string(agent_index)+"/cmd_vel", 1,
            boost::bind(&RosEnvironment::process_action, this, agent_index, _1));
        action_subscribers.push_back(action_subscriber);
    }

    ROS_INFO("Initialized environment");
}

mapf_environment::Observation RosEnvironment::convert_observation(Observation obs)
{
    mapf_environment::Observation ros_obs;

    ros_obs.agent_pose.x = obs.agent_pose.x;
    ros_obs.agent_pose.y = obs.agent_pose.y;
    ros_obs.agent_pose.z = obs.agent_pose.z;

    ros_obs.agent_twist.x = obs.agent_twist.x;
    ros_obs.agent_twist.y = obs.agent_twist.y;
    ros_obs.agent_twist.z = obs.agent_twist.z;

    ros_obs.goal_pose.x = obs.goal_pose.x;
    ros_obs.goal_pose.y = obs.goal_pose.y;
    ros_obs.goal_pose.z = obs.goal_pose.z;

    ros_obs.scan.ranges = obs.scan;
    ros_obs.reward = obs.reward;

    return ros_obs;
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

    EnvStep env_step = env->step(coll_action);
    std::string info;
    info += "Rewards: ";
    for (auto obs : env_step.observations)
    {
        info += "  " + std::to_string(obs.reward);
    }

    std::cout << info << std::endl;

    cv_bridge::CvImage cv_img;
    cv_img.header.stamp = ros::Time::now();
    cv_img.encoding = "bgr8";
    cv_img.image = env->get_rendered_pic();
    render_publisher.publish(cv_img.toImageMsg());

    for (int agent_index=0; agent_index < env->get_number_of_agents(); agent_index++)
    {
        mapf_environment::Observation obs = convert_observation(env_step.observations[agent_index]);
        observation_publishers[agent_index].publish(obs);
    }
}

void RosEnvironment::process_action(int agent_index, const geometry_msgs::TwistConstPtr& action)
{
    if (std::abs(action->linear.x) < 1.)
        coll_action[agent_index].x = action->linear.x;
    else
    {
        ROS_WARN_STREAM("Received linear velocity greater than 1 m/s ("
            << action->linear.x
            << "), trimming");
        if (action->linear.x > 0)
        {
            coll_action[agent_index].x = 1.;
        }
        else
            coll_action[agent_index].x = -1;
    }

    if (std::abs(action->angular.z) < M_PI/2)
        coll_action[agent_index].z = action->angular.z;
    else
    {
        ROS_WARN_STREAM("Received angular velocity greater than "
            << M_PI/2
            << " rad/s ("
            << action->angular.z
            << "), trimming");
        if (action->angular.z > 0)
        {
            coll_action[agent_index].z = M_PI/2.;
        }
        else
            coll_action[agent_index].z = -M_PI/2;
    }
}
