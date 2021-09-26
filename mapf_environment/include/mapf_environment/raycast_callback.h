// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_RAYCAST_CALLBACK_H
#define MAPF_ENVIRONMENT_RAYCAST_CALLBACK_H

#include <box2d/box2d.h>

/*! \brief Raycast callback class for box2d
 *
 * Based on: https://www.iforce2d.net/b2dtut/world-querying
 */
class RayCastClosestCallback : public b2RayCastCallback
{
public:
    bool hit;
    b2Vec2 point;
    b2Vec2 normal;

    RayCastClosestCallback();
    float ReportFixture(b2Fixture* fixture, const b2Vec2& point, const b2Vec2& normal, float fraction);
};

#endif  // MAPF_ENVIRONMENT_RAYCAST_CALLBACK_H
