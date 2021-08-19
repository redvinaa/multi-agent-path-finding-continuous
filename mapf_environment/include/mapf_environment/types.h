// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_TYPES_H
#define MAPF_ENVIRONMENT_TYPES_H

#include <vector>
#include "geometry_msgs/Twist.h"
#include <mapf_environment/Observation.h>
#include <mapf_environment/EnvStep.h>

using Value                 = float;

using Observation           = mapf_environment::Observation;
using CollectiveObservation = std::vector<mapf_environment::Observation>;

using Action                = geometry_msgs::Twist;
using CollectiveAction      = std::vector<geometry_msgs::Twist>;

using EnvStep               = mapf_environment::EnvStep;

#endif  // MAPF_ENVIRONMENT_TYPES_H
