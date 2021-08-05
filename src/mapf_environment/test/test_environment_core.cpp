#include "mapf_environment/environment_core.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <ros/package.h>
#include <gtest/gtest.h>

TEST(EnvironmentCore, constructorRuns) {
	try {
		std::string pkg_path = ros::package::getPath("mapf_environment");
		std::string image_path = pkg_path + "/maps/test_4x4.jpg";
		Environment env(image_path);
	} catch (const std::exception &e) {
		ADD_FAILURE() << "Exception in constructor: " << e.what();
	}
}

class EnvironmentFixture : public testing::Test {
	protected:
		Environment* environment;

		void SetUp() override {
			std::string pkg_path = ros::package::getPath("mapf_environment");
			std::string image_path = pkg_path + "/maps/test_4x4.jpg";
			environment = new Environment(image_path);
		}

		void TearDown() override {
			delete environment;
		}
};

TEST_F(EnvironmentFixture, testConstructor) {
	EXPECT_FALSE(environment->map_image_raw.empty());
	EXPECT_TRUE(environment->robot_radius != 0);
	EXPECT_TRUE(environment->map_image.size().height == environment->render_height);
}

TEST_F(EnvironmentFixture, testMap) {
	EXPECT_EQ(environment->map_image_raw.size(), cv::Size(4, 4));
	EXPECT_EQ(environment->map_image_raw.channels(), 1);
	EXPECT_TRUE(environment->map_image_raw.at<uchar>(0, 0) > 125);
	EXPECT_TRUE(environment->map_image_raw.at<uchar>(1, 0) < 125);
	EXPECT_EQ(environment->map_image.channels(), 3);
	EXPECT_EQ(environment->map_image.size().height, environment->render_height);

	// cv::Mat map = environment->render();
	// cv::imshow("test_4x4.jpg", map);
	// cv::waitKey(0);
}

TEST_F(EnvironmentFixture, testPhysics) {
	bool hit;

	// test that nothing is in the upper left cell
	b2Transform transform;
	b2Vec2 point;

	point.Set(0.5f, 3.5f);
	hit = false;
	for (auto it=environment->obstacle_bodies.begin(); it!=environment->obstacle_bodies.end(); it++) {
		auto body = *it;
		transform = body->GetTransform();

		auto shape = body->GetFixtureList()->GetShape();
		ASSERT_TRUE(shape != NULL);

		hit = shape->TestPoint(transform, point);
		if (hit == true)
			break;
	}
	EXPECT_FALSE(hit);

	// test that there is an obstacle right below that

	point.Set(0.5f, 2.5f);
	hit = false;
	for (auto it=environment->obstacle_bodies.begin(); it!=environment->obstacle_bodies.end(); it++) {
		b2Body* body = *it;
		transform = body->GetTransform();

		b2Fixture* fixture = body->GetFixtureList();
		b2Shape* shape = fixture->GetShape();
		ASSERT_TRUE(shape != NULL);

		hit = shape->TestPoint(transform, point);
		if (hit == true)
			break;
	}
	EXPECT_TRUE(hit);
}

TEST_F(EnvironmentFixture, testAddAndRemoveAgent) {
	EXPECT_EQ(environment->number_of_agents, 0);
	int idx = environment->add_agent();
	EXPECT_EQ(idx, 0);
	EXPECT_EQ(environment->number_of_agents, 1);

	EXPECT_EQ(1, environment->agent_bodies.size());
	EXPECT_EQ(1, environment->goal_positions.size());
	EXPECT_EQ(1, environment->agent_colors.size());
	EXPECT_EQ(1, environment->agent_lin_vel.size());
	EXPECT_EQ(1, environment->agent_ang_vel.size());
	EXPECT_EQ(1, environment->collisions.size());
	EXPECT_EQ(1, environment->laser_scans.size());

	EXPECT_EQ(environment->laser_nrays, environment->laser_scans[0].ranges.size());

	environment->remove_agent(0);
	EXPECT_EQ(environment->number_of_agents, 0);

	EXPECT_EQ(0, environment->agent_bodies.size());
	EXPECT_EQ(0, environment->goal_positions.size());
	EXPECT_EQ(0, environment->agent_colors.size());
	EXPECT_EQ(0, environment->agent_lin_vel.size());
	EXPECT_EQ(0, environment->agent_ang_vel.size());
	EXPECT_EQ(0, environment->collisions.size());
	EXPECT_EQ(0, environment->laser_scans.size());
}

TEST(box2d, shape_hit) {
	b2PolygonShape shape;
	shape.SetAsBox(0.5f, 0.5f);

	b2Transform transform;
	transform.SetIdentity();
	b2Vec2 point(0.0f, 0.0f);

	bool hit = shape.TestPoint(transform, point);
	EXPECT_TRUE(hit);
}

int main(int argc, char **argv){
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
