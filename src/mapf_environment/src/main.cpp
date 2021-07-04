#include <ros/ros.h>
#include "mapf_environment/environment.h"

int main(int argc, char** argv) {
	ros::init(argc, argv, "environment");
	ros::NodeHandle nh;

	Environment env(nh);
	ros::spin();
}
