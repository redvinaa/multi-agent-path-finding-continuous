#include <mapf_environment/raycast_callback.h>

RayCastClosestCallback::RayCastClosestCallback()
{
    hit = false;
}

float RayCastClosestCallback::ReportFixture(b2Fixture* fixture, const b2Vec2& point, const b2Vec2& normal, float fraction)
{
    b2Body* body = fixture->GetBody();

    hit = true;
    this->point = point;
    this->normal = normal;

    // By returning the current fraction, we instruct the calling code to clip the ray and
    // continue the ray-cast to the next fixture. WARNING: do not assume that fixtures
    // are reported in order. However, by clipping, we can always get the closest fixture.
    return fraction;
}
