#include <box2d/box2d.h>

class RayCastClosestCallback : public b2RayCastCallback
{
public:
	bool hit;
	b2Vec2 point;
	b2Vec2 normal;
	
	RayCastClosestCallback();
	float ReportFixture(b2Fixture* fixture, const b2Vec2& point, const b2Vec2& normal, float fraction);
};
