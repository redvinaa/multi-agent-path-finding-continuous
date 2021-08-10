// ros headers
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <geometry_msgs/Twist.h>
#include <mapf_environment/AddAgent.h>
#include <mapf_environment/Observation.h>
#include <mapf_environment/Action.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/LaserScan.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>

// other headers
#include <box2d/box2d.h>
#include <vector>
#include <string>


class RosEnvironment {
	private:
		// ros fields
		ros::NodeHandle nh;
		std::vector<ros::Publisher> observation_publishers;
		std::vector<ros::Subscriber> action_subscribers;
		ros::ServiceServer add_agent_service;
		ros::Timer physics_timer, render_timer, publish_observations_timer, count_call_agents_timer;
		image_transport::ImageTransport it;
		image_transport::Publisher render_publisher;
		ddynamic_reconfigure::DDynamicReconfigure ddr;

		std::vector<bool> action_received; // check if action is received before publishing next observation

		bool add_agent(mapf_environment::AddAgentRequest& req, mapf_environment::AddAgentResponse& res);
		bool remove_agent(mapf_environment::RemoveAgentRequest& req, mapf_environment::RemoveAgentResponse& res);
		void step_physics(const ros::TimerEvent&);
		void render(const ros::TimerEvent&);
		void publish_observations(const ros::TimerEvent&);
		void process_action(int agent_index, const mapf_environment::ActionConstPtr& action);
		void count_call_agents_callback(const ros::TimerEvent&);
		mapf_environment::Observation get_observation(int agent_index);

		// ddr callbacks
		void ddr_physics_frequency(double frequency);
		void ddr_render_frequency(double frequency);
		void ddr_agent_service_frequency(double frequency);

	public:
		Environment(ros::NodeHandle _nh);
};
