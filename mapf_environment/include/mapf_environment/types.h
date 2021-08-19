#pragma once

#include "geometry_msgs/Twist.h"
#include <mapf_environment/Observation.h>
#include <mapf_environment/EnvStep.h>

using Value                 = float;

using Observation           = mapf_environment::Observation;
using CollectiveObservation = std::vector<mapf_environment::Observation>;

using Action                = geometry_msgs::Twist;
using CollectiveAction      = std::vector<geometry_msgs::Twist>;

using EnvStep               = mapf_environment::EnvStep;
