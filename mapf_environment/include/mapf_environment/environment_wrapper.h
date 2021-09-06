// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_ENVIRONMENT_WRAPPER_H
#define MAPF_ENVIRONMENT_ENVIRONMENT_WRAPPER_H

// ros headers
#include "ros/ros.h"
#include "std_srvs/Empty.h"
#include "geometry_msgs/Twist.h"
#include "mapf_environment/AddAgent.h"
#include "mapf_environment/RemoveAgent.h"
#include "mapf_environment/Observation.h"
#include "sensor_msgs/LaserScan.h"
#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"

// other headers
#include "mapf_environment/environment.h"
#include <vector>
#include <string>
#include <memory>


class RosEnvironment
{
    private:
        // ros fields
        ros::NodeHandle nh;
        std::vector<ros::Publisher> observation_publishers;
        std::vector<ros::Subscriber> action_subscribers;
        ros::ServiceServer add_agent_service, remove_agent_service;
        ros::Timer physics_timer;
        image_transport::ImageTransport it;
        image_transport::Publisher render_publisher;
        ros::Publisher sim_time_publisher;

        std::shared_ptr<Environment> env;
        std::string map_path;
        CollectiveAction coll_action;

        /*! \brief Call add_agent in Environment, and set up ros communication */
        bool add_agent(mapf_environment::AddAgentRequest& req, mapf_environment::AddAgentResponse& res);

        /*! \brief Remove agent from Environment, and delete ros communication */
        bool remove_agent(mapf_environment::RemoveAgentRequest& req, mapf_environment::RemoveAgentResponse& res);

        /*! \brief Convert from struct Observation to mapf_environment::Observation (ROS) */
        static mapf_environment::Observation convert_observation(Observation obs);

        /*! \brief Step Environment, publish observations, sim_time and rendered image */
        void step(const ros::TimerEvent&);

        /*! \brief Save action for given agent, that will be passed to the Envitonment */
        void process_action(int agent_index, const geometry_msgs::TwistConstPtr& action);

    public:
        /*! \brief Constructor, adds one agent by default */
        explicit RosEnvironment(ros::NodeHandle _nh);
};

#endif  // MAPF_ENVIRONMENT_ENVIRONMENT_WRAPPER_H
