// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_TYPES_H
#define MAPF_ENVIRONMENT_TYPES_H

#include <vector>

using Value                 = float;

/*! \brief Stores a 3D point: x, y, z */
struct Point
{
    float x, y, z;
};

/*! \brief Vector to store ranges of the lidar */
using LaserScan = std::vector<float>;

struct Observation
{
    /*! \brief x and y stores the position, z the angle of an agent */
    Point agent_pose;

    /*! \brief x stores the linear, z the angular velocity */
    Point agent_twist;

    /*! \brief Only x and y is used, as position */
    Point goal_pose;

    /*! \brief Vector to store ranges of the lidar */
    LaserScan scan;

    /*! \brief Reward received after the previous step */
    float reward;
};

using CollectiveObservation = std::vector<Observation>;

using Action                = Point;
using CollectiveAction      = std::vector<Action>;

/*! \brief Data returned from the Environment after calling step */
struct EnvStep
{
    /*! \brief Observations of all the agents */
    CollectiveObservation observations;

    /*! \brief Whether the episode is done */
    bool done;
};

#endif  // MAPF_ENVIRONMENT_TYPES_H
