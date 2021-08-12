#include <ros/ros.h>
#include "mapf_environment/environment_wrapper.h"

int main(int argc, char** argv)
{
    ros::init(argc, argv, "mapf_environment");
    ros::NodeHandle nh;

    RosEnvironment env(nh);
    ros::spin();
}
