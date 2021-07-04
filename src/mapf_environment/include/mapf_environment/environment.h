// ros headers
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <mapf_environment/Observation.h>
#include <geometry_msgs/Twist.h>
#include <mapf_environment/AddAgent.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/types.hpp>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>

// other headers
#include <box2d/box2d.h>
#include <vector>
#include <string>


class Environment {
	private:
		// ros fields
		ros::NodeHandle nh;
		std::vector<ros::Publisher> observation_publishers; // debug only
		std::vector<ros::Subscriber> action_subscribers; // debug only
		std::vector<ros::ServiceClient> call_agent_services;
		ros::ServiceServer add_agent_service;
		ros::Timer physics_timer, render_timer, agent_service_timer, count_call_agents_timer;
		image_transport::ImageTransport it;
		image_transport::Publisher render_publisher;
		ddynamic_reconfigure::DDynamicReconfigure ddr;

		// box2d fields
		b2Vec2  gravity;
		b2World world;
		std::vector<b2Body*> agent_bodies;
		std::vector<b2Body*> obstacle_bodies;
		std::vector<b2Vec2> goal_positions;

		// physics properties
		double physics_frequency, physics_step_size;
		int32 velocity_iterations, position_iterations;

		// other fields
		std::vector<cv::viz::Color> agent_colors;
		std::vector<float> agent_lin_vel;
		std::vector<float> agent_ang_vel;
		std::vector<sensor_msgs::LaserScan> laser_scans;
		bool draw_laser, physics_on, rendering_on;
		double render_frequency, block_size, scale_factor, laser_max_angle, laser_max_dist,
			robot_diam, robot_radius, map_width, map_height, agent_service_frequency;
		std::string map_path;
		int render_height, laser_nrays, count_call_agents;
		cv::Mat map_image_raw, map_image, rendered_image;
		cv::viz::Color color;

		// private methods
		int generate_empty_index(void);
		void init_physics(void);
		void init_map(void);
		bool add_agent(mapf_environment::AddAgentRequest& req, mapf_environment::AddAgentResponse& res);
		void step_physics(const ros::TimerEvent&);
		void render(const ros::TimerEvent&);
		void action_callback(const geometry_msgs::TwistConstPtr& msg, const int agent_index);
		void call_agent(const ros::TimerEvent&);
		void count_call_agents_callback(const ros::TimerEvent&);

		// ddr callbacks
		void ddr_physics_frequency(double frequency);
		void ddr_render_frequency(double frequency);
		void ddr_agent_service_frequency(double frequency);

	public:
		Environment(ros::NodeHandle _nh);
};
