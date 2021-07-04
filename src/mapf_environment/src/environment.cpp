// ros headers
#include <ros/ros.h>
#include <std_srvs/Empty.h>
#include <mapf_environment/Observation.h>
#include <mapf_environment/AddAgent.h>
#include <geometry_msgs/Twist.h>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/viz/types.hpp>
#include <ddynamic_reconfigure/ddynamic_reconfigure.h>
#include <mapf_environment/CallAgent.h>

// other headers
#include <mapf_environment/environment.h>
#include <mapf_environment/raycast_callback.h>
#include <box2d/box2d.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <stdlib.h>
#include <algorithm>
#include <cmath>

std::string type2str(int type) {
  std::string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

Environment::Environment(ros::NodeHandle _nh):
		nh(_nh), gravity(0, 0), world(gravity), it(_nh), rendering_on(true), physics_on(true) {
	// read parameters
	ros::NodeHandle nh_private("~");
	if (!nh_private.getParam("map_path", map_path)) throw std::runtime_error("Missing parameter: map_path");
	nh_private.param<double>("physics_frequency",        physics_frequency,        100);
	nh_private.param<double>("physics_step_size",        physics_step_size,        0.01);
	nh_private.param<double>("render_frequency",         render_frequency,         30);
	nh_private.param<double>("laser_max_angle",          laser_max_angle,          45*M_PI/180);
	nh_private.param<double>("laser_max_dist",           laser_max_dist,           10);
	nh_private.param<double>("robot_diam",               robot_diam,               0.8);
	nh_private.param<double>("agent_service_frequency",  agent_service_frequency,  10);
	nh_private.param<int>("velocity_iterations",         velocity_iterations,      6);
	nh_private.param<int>("position_iterations",         position_iterations,      2);
	nh_private.param<int>("render_height",               render_height,            700);
	nh_private.param<int>("laser_nrays",                 laser_nrays,              10);
	nh_private.param<bool>("draw_laser",                 draw_laser,               false);

	// dynamic reconfigure
	ddr.registerVariable<double>("laser_max_angle", &laser_max_angle, "Maximum (half) angle of the raycast sensor", 0, M_PI/2);
	ddr.registerVariable<double>("laser_max_dist",  &laser_max_dist,  "Maximum distance of the raycast sensor", 0.1, 20);
	ddr.registerVariable<bool> ("physics_on",       &physics_on,      "Start or stop physics", false, true);
	ddr.registerVariable<bool> ("rendering_on",     &rendering_on,    "Start or stop rendering", false, true);
	ddr.registerVariable<bool> ("draw_laser",       &draw_laser,      "Visualize raycast sensor", false, true);
	ddr.publishServicesTopics();

	// initialize ros communication
	render_publisher = it.advertise("image", 10);
	add_agent_service = nh.advertiseService("add_agent", &Environment::add_agent, this);

	// timers
	physics_timer       = nh.createTimer(ros::Duration(1/physics_frequency),       &Environment::step_physics, this);
	render_timer        = nh.createTimer(ros::Duration(1/render_frequency),        &Environment::render,       this);
	agent_service_timer = nh.createTimer(ros::Duration(1/agent_service_frequency), &Environment::call_agent,   this);

	srand(time(NULL));
	robot_radius = robot_diam/2;
	init_map(); // load map before physics
	init_physics();
	ROS_INFO("Multi-agent path finding environment initialized, add agents and start physics");
}

void Environment::init_map(void) {
	map_image_raw = cv::imread(map_path, cv::IMREAD_GRAYSCALE);
	ROS_INFO_STREAM("Map type: " << type2str(map_image_raw.type()));
	if (map_image_raw.empty()) throw std::runtime_error("Map image not found: " + map_path);

	map_width = map_image_raw.size().width;
	map_height = map_image_raw.size().height;
	scale_factor = (float)render_height / map_height;

	cv::resize(map_image_raw, map_image, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
	cv::cvtColor(map_image, map_image, cv::COLOR_GRAY2BGR);
	map_image.copyTo(rendered_image); // set size of rendered image
}

int Environment::generate_empty_index() {
	int num_index = map_image_raw.size().width * map_image_raw.size().height; // number of (flattened) indices on the grid map
	std::vector<int> map_indices;
	map_indices.resize(num_index);
	for (int i=0; i<num_index; i++) {
		map_indices[i] = i;
	}

	std::random_shuffle(map_indices.begin(), map_indices.end());

	for (auto it = map_indices.begin(); it!=map_indices.end(); it++) {
		int index = *it;

		// get position from index
		int row = index / map_image_raw.size().width;
		int col = index - row * map_image_raw.size().width; // integer division
		float x_pos = col + 0.5; // center of cell
		float y_pos = map_height - row - 0.5;
		b2Vec2 index_position(x_pos, y_pos);
		ROS_INFO_STREAM("Index position: x=" << x_pos << ", y=" << y_pos);

		// check if cell has a wall
		ROS_INFO_STREAM("Map value there: " << (int)map_image_raw.at<unsigned char>(row, col));
		if ((int)map_image_raw.at<unsigned char>(row, col) < 255/2) continue;

		// check if any agent is near
		bool somethings_near = false;
		for (int agent=0; agent<agent_bodies.size(); agent++) {
			ROS_INFO_STREAM("Checking for agent " << agent);
			if ((agent_bodies[agent]->GetPosition() - index_position).Length() < robot_diam) {
				somethings_near = true;
				break;
			}
			if ((goal_positions[agent] - index_position).Length() < robot_diam) {
				somethings_near = true;
				break;
			}
		}
		if (!somethings_near) return index;
	}
	// no index found, throw error
	throw std::runtime_error("No position could be generated");
}

void Environment::init_physics(void) {
	// create bounding box, (0, 0) is the bottom left corner
	b2BodyDef bd;
	bd.position.Set(map_width/2, map_height/2);
	b2Body* ground = world.CreateBody(&bd);

	b2EdgeShape shape;

	b2FixtureDef sd;
	sd.shape = &shape;
	sd.density = 0;

	// Left vertical
	shape.SetTwoSided(b2Vec2(-map_width/2, -map_height/2), b2Vec2(-map_width/2, map_height/2));
	ground->CreateFixture(&sd);

	// Right vertical
	shape.SetTwoSided(b2Vec2(map_width/2, -map_height/2), b2Vec2(map_width/2, map_height/2));
	ground->CreateFixture(&sd);

	// Top horizontal
	shape.SetTwoSided(b2Vec2(-map_width/2, map_height/2), b2Vec2(map_width/2, map_height/2));
	ground->CreateFixture(&sd);

	// Bottom horizontal
	shape.SetTwoSided(b2Vec2(-map_width/2, -map_height/2), b2Vec2(map_width/2, -map_height/2));
	ground->CreateFixture(&sd);

	// create static obstacles
	for(int row = 0; row < map_image_raw.rows; ++row) {
		uchar* p = map_image_raw.ptr(row);
		for(int col = 0; col < map_image_raw.cols; ++col) {
			int pixel_value = *p;
			if (pixel_value < 255/2) {
				// add obstacle
				b2BodyDef bd;
				bd.type = b2_dynamicBody;
				float x_pos = col + 0.5;
				float y_pos = map_height - row - 0.5;
				ROS_DEBUG_STREAM("Adding obstacle at x=" << x_pos << ", y=" << y_pos);
				bd.position.Set(x_pos, y_pos);
				b2Body* obst = world.CreateBody(&bd);

				b2PolygonShape boxShape;
				boxShape.SetAsBox(0.5f, 0.5f);

				b2FixtureDef boxFixtureDef;
				boxFixtureDef.shape = &boxShape;
				obst->CreateFixture(&boxFixtureDef);
				b2Vec2 point= obst->GetPosition();

				obstacle_bodies.push_back(obst);
			}
			*p++; //points to each pixel value in turn assuming a CV_8UC1 greyscale image 
		}
	}   
}

bool Environment::add_agent(mapf_environment::AddAgentRequest& req, mapf_environment::AddAgentResponse& res) {
	// create simulated body
	b2BodyDef bodyDef;
	bodyDef.type = b2_dynamicBody;

	// generate random starting position
	int index = generate_empty_index();
	int row = index / map_image_raw.size().width;
	int col = index - row * map_image_raw.size().width; // integer division
	float x_pos = col + 0.5;
	float y_pos = map_height - row - 0.5;
	bodyDef.position.Set(x_pos, y_pos);
	b2Body* body = world.CreateBody(&bodyDef);

	// generate random goal position
	index = generate_empty_index();
	row = index / map_image_raw.size().width;
	col = index - row * map_image_raw.size().width; // integer division
	x_pos = col + 0.5;
	y_pos = map_height - row - 0.5;
	goal_positions.push_back(b2Vec2(x_pos, y_pos));

	b2CircleShape circleShape;
	circleShape.m_p.Set(0, 0);
	circleShape.m_radius = robot_radius;

	b2FixtureDef fixtureDef;
	fixtureDef.shape = &circleShape;
	fixtureDef.density = 1.0f;
	fixtureDef.friction = 0;
	fixtureDef.restitution = 0;

	body->CreateFixture(&fixtureDef);
	agent_bodies.push_back(body);

	// generate random color for agent
	int b = (float)rand() / (float)RAND_MAX * 255;
	int r = (float)rand() / (float)RAND_MAX * 255;
	int g = (float)rand() / (float)RAND_MAX * 255;
	cv::viz::Color agent_color(b, r, g);
	agent_colors.push_back(agent_color);

	// create ros communication
	const int agent_index = action_subscribers.size();

	boost::function<void(const boost::shared_ptr<geometry_msgs::Twist const>&)> callback =
		boost::bind(&Environment::action_callback, this, _1, agent_index);
	ros::Subscriber action_subscriber = nh.subscribe<geometry_msgs::Twist>("agent_"+std::to_string(agent_index)+"/cmd_vel", 1000, callback);
	action_subscribers.push_back(action_subscriber); // debug only

	ros::Publisher observation_publisher = nh.advertise<mapf_environment::Observation>("agent_"+std::to_string(agent_index)+"/observation", 1000);
	observation_publishers.push_back(observation_publisher); // debug only

	call_agent_services.push_back(nh.serviceClient<mapf_environment::CallAgent>("agent_"+std::to_string(agent_index)+"/call_agent"));

	// fill laser_scans
	sensor_msgs::LaserScan scan;
	scan.ranges.resize(laser_nrays);
	laser_scans.push_back(scan);
	
	agent_lin_vel.push_back(0);
	agent_ang_vel.push_back(0);
	ROS_INFO_STREAM("Added agent " << agent_index);
	res.success = true;
	res.agent_index = agent_index;
	return true;
}

void Environment::action_callback(const geometry_msgs::TwistConstPtr& msg, const int agent_index) {
	agent_lin_vel[agent_index] = msg->linear.x;
	agent_ang_vel[agent_index] = msg->angular.z;
}

void Environment::call_agent(const ros::TimerEvent&) {
	for (int agent=0; agent<call_agent_services.size(); agent++) {
		ros::ServiceClient client = call_agent_services[agent];
		client.waitForExistence();

		mapf_environment::CallAgent data;
		// TODO fill with observation
		if (!client.call(data)) {
			throw std::runtime_error("Communication to agent "+std::to_string(agent)+" failed");
		}
	}
}


void Environment::step_physics(const ros::TimerEvent&) {
	if (physics_on == false) return;

	world.Step(physics_step_size, velocity_iterations, position_iterations);
	for (int i=0; i<agent_bodies.size(); i++) {
		b2Body* agent = agent_bodies[i];
		float angle = agent->GetAngle();

		agent->SetLinearVelocity(b2Vec2(agent_lin_vel[i]*std::cos(angle), agent_lin_vel[i]*std::sin(angle)));
		agent->SetAngularVelocity(agent_ang_vel[i]);

		// check if goal is reached
		if ((agent->GetPosition() - goal_positions[i]).Length() < robot_diam) {
			// goal is reached
			int index = generate_empty_index();
			float row = index / map_image_raw.size().width;
			float col = index - row * map_image_raw.size().width; // integer division
			float x_pos = col + 0.5;
			float y_pos = map_height - row - 0.5;
			ROS_INFO_STREAM("Index position: x=" << x_pos << ", y=" << y_pos);

			goal_positions[i] = b2Vec2(x_pos, y_pos);
		}

		// calculate laser scans
		b2Vec2 pt_from, pt_to;
		b2Vec2 position = agent->GetPosition();

		pt_from.x = robot_radius * std::cos(angle) + position.x;
		pt_from.y = robot_radius * std::sin(angle) + position.y;
		for (int j=0; j<laser_nrays; j++) {
			float laser_angle = angle - laser_max_angle + j * laser_max_angle * 2 / laser_nrays;
			pt_to.x = laser_max_dist * std::cos(laser_angle) + pt_from.x;
			pt_to.y = laser_max_dist * std::sin(laser_angle) + pt_from.y;

			RayCastClosestCallback callback;
			world.RayCast(&callback, pt_from, pt_to);

			float dist = laser_max_dist;
			if (callback.hit) {
				dist = (pt_from - callback.point).Length();
			}
			laser_scans[i].ranges[j] = dist;
		}
	}
}

void Environment::render(const ros::TimerEvent&) {
	if (!rendering_on) return;

	rendered_image = color.white();

	// draw obstacles
	for (int i=0; i<obstacle_bodies.size(); i++) {
		b2Vec2 pos = obstacle_bodies[i]->GetPosition();
		cv::Point pt1((pos.x-0.5)*scale_factor, render_height - (pos.y-0.5)*scale_factor);
		cv::Point pt2((pos.x+0.5)*scale_factor, render_height - (pos.y+0.5)*scale_factor);
		cv::rectangle(rendered_image, pt1, pt2, color.black(), -1);
	}

	// draw goals
	for (int i=0; i<goal_positions.size(); i++) {
		b2Vec2 position = goal_positions[i];

		// big circle
		cv::Point center(position.x*scale_factor, render_height-position.y*scale_factor);
		cv::circle(rendered_image, center, robot_radius*scale_factor, agent_colors[i], -1);
	}

	// draw agents
	float inner_radius = robot_radius * 0.8;
	auto font = cv::FONT_HERSHEY_TRIPLEX;
	int thickness = 3;
	int base_line = 0;
	auto scale = cv::getFontScaleFromHeight(font, inner_radius*scale_factor, thickness);
	for (int i=0; i<agent_bodies.size(); i++) {
		b2Vec2 position = agent_bodies[i]->GetPosition();
		float angle = agent_bodies[i]->GetAngle();

		// big circle
		cv::Point center(position.x*scale_factor, render_height-position.y*scale_factor);
		cv::circle(rendered_image, center, robot_radius*scale_factor, agent_colors[i], -1);

		// arrow
		cv::Point triangle[1][3];
		triangle[0][0].x = robot_radius * scale_factor * std::cos(-angle) + center.x;
		triangle[0][0].y = robot_radius * scale_factor * std::sin(-angle) + center.y;
		triangle[0][1].x = 0.9*inner_radius * scale_factor * std::cos(M_PI/2 -angle) + center.x;
		triangle[0][1].y = 0.9*inner_radius * scale_factor * std::sin(M_PI/2 -angle) + center.y;
		triangle[0][2].x = 0.9*inner_radius * scale_factor * std::cos(-M_PI/2 -angle) + center.x;
		triangle[0][2].y = 0.9*inner_radius * scale_factor * std::sin(-M_PI/2 -angle) + center.y;

		const cv::Point* ppt[1] = { triangle[0] };
		int npt[] = {3};
		cv::fillPoly(rendered_image, ppt, npt, 1, color.white());

		// small circle
		cv::circle(rendered_image, center, (int)(inner_radius*scale_factor), color.white(), -1);

		// number
		cv::Point textSize = cv::getTextSize(std::to_string(i), font, scale, thickness, &base_line);
		textSize.x = center.x - textSize.x / 2;
		textSize.y = center.y + textSize.y / 2;
		cv::putText(rendered_image, std::to_string(i), textSize, font, scale, color.black(), thickness);

		// laser scans
		if (draw_laser) {
			cv::Point pt_from, pt_to;
			pt_from.x = robot_radius * scale_factor * std::cos(-angle) + center.x;
			pt_from.y = robot_radius * scale_factor * std::sin(-angle) + center.y;
			for (int j=0; j<laser_nrays; j++) {
				float laser_angle = angle - laser_max_angle + j * laser_max_angle * 2 / laser_nrays;
				pt_to.x = laser_scans[i].ranges[j] * scale_factor * std::cos(-laser_angle) + pt_from.x;
				pt_to.y = laser_scans[i].ranges[j] * scale_factor * std::sin(-laser_angle) + pt_from.y;
				cv::line(rendered_image, pt_from, pt_to, color.gray());
			}
		}
	}

	cv_bridge::CvImagePtr cv_ptr(new cv_bridge::CvImage);
	cv_ptr->header.stamp = ros::Time::now();
	cv_ptr->encoding = "bgr8";
	cv_ptr->image = rendered_image;
	render_publisher.publish(cv_ptr->toImageMsg());
}
