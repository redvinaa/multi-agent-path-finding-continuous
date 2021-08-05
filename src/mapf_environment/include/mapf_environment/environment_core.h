// ros datatype headers
#include <geometry_msgs/Twist.h>
#include <mapf_environment/Observation.h>
#include <sensor_msgs/LaserScan.h>

// other headers
#include <box2d/box2d.h>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>


class Environment {
	public:
		// box2d fields
		b2Vec2  gravity;
		b2World world;
		std::vector<b2Body*> agent_bodies;
		std::vector<b2Body*> obstacle_bodies;
		std::vector<b2Vec2> goal_positions;

		// physics properties
		double physics_step_size;
		int32 velocity_iterations, position_iterations;

		// other fields
		std::vector<cv::Scalar> agent_colors;
		std::vector<float> agent_lin_vel, agent_ang_vel;
		std::vector<bool> collisions; // TODO collisions not implemented yet
		std::vector<sensor_msgs::LaserScan> laser_scans;
		bool draw_laser;
		float block_size, scale_factor, laser_max_angle, laser_max_dist,
			robot_diam, robot_radius, map_width, map_height, 
			goal_reaching_reward, collision_reward, step_reward;
		std::string map_path;
		int render_height, laser_nrays, number_of_agents;
		cv::Mat map_image_raw, map_image, rendered_image;
		cv::Scalar color;

	public:
		bool done;

		Environment(std::string _map_path,
			float _physics_step_size=0.01,
			float _laser_max_angle=45.*M_PI/180.,
			float _laser_max_dist=10.,
			float _robot_diam=0.8,
			int _velocity_iterations=6,
			int _position_iterations=2,
			int _render_height=700,
			int _laser_nrays=10,
			bool _draw_laser=false,
			float _goal_reaching_reward=0.,
			float _collision_reward=-1.,
			float _step_reward=-1.);

		void init_map(void);
		void init_physics(void);
		int generate_empty_index(void);
		void reset();
		int add_agent();
		void remove_agent(int agent_index);
		void step_physics();
		cv::Mat render();
		void process_action(int agent_index, geometry_msgs::Twist action);
		mapf_environment::Observation get_observation(int agent_index);
};
