// Copyright 2021 Reda Vince

// ros datatype headers
#include <geometry_msgs/Twist.h>
#include <mapf_environment/Observation.h>
#include <mapf_environment/EnvStep.h>
#include <sensor_msgs/LaserScan.h>

// other headers
#include <mapf_environment/environment.h>
#include <mapf_environment/raycast_callback.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <box2d/box2d.h>
#include <vector>
#include <string>
#include <stdexcept>
#include <functional>
#include <stdlib.h>
#include <algorithm>
#include <cmath>


Environment::Environment(std::string _map_path,
    float        _physics_step_size /* 0.01 */,
    int          _step_multiply /* 5 */,
    float        _laser_max_angle /* 45.*M_PI/180. */,
    float        _laser_max_dist /* 10. */,
    float        _robot_diam /* 0.8 */,
    int          _velocity_iterations /* 6 */,
    int          _position_iterations /* 2 */,
    int          _render_height /* 700 */,
    int          _laser_nrays /* 10 */,
    bool         _draw_laser /* false */,
    float        _goal_reaching_reward /* 0. */,
    float        _collision_reward /* -1. */,
    float        _step_reward /* -1. */,
    unsigned int _seed /* 0 */):
        gravity(0, 0),
        world(gravity),
        number_of_agents(0),
        done(true)
{
    map_path             = _map_path;
    physics_step_size    = _physics_step_size;
    step_multiply        = _step_multiply;
    laser_max_angle      = _laser_max_angle;
    laser_max_dist       = _laser_max_dist;
    robot_diam           = _robot_diam;
    velocity_iterations  = _velocity_iterations;
    position_iterations  = _position_iterations;
    render_height        = _render_height;
    laser_nrays          = _laser_nrays;
    draw_laser           = _draw_laser;
    goal_reaching_reward = _goal_reaching_reward;
    collision_reward     = _collision_reward;
    step_reward          = _step_reward;
    seed                 = _seed;

    robot_radius = robot_diam/2;
    init_map();  // load map before physics
    init_physics();
}

void Environment::init_map(void)
{
    map_image_raw = cv::imread(map_path, cv::IMREAD_GRAYSCALE);
    if (map_image_raw.empty()) throw std::runtime_error("Map image not found: '" + map_path + "'");

    map_width = map_image_raw.size().width;
    map_height = map_image_raw.size().height;
    scale_factor = static_cast<float>(render_height) / map_height;

    cv::resize(map_image_raw, map_image, cv::Size(), scale_factor, scale_factor, cv::INTER_AREA);
    cv::cvtColor(map_image, map_image, cv::COLOR_GRAY2BGR);
    map_image.copyTo(rendered_image);  // set size of rendered image
}

void Environment::init_physics(void)
{
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
    for (int row = 0; row < map_image_raw.rows; ++row)
    {
        uchar* p = map_image_raw.ptr(row);
        for (int col = 0; col < map_image_raw.cols; ++col)
        {
            int pixel_value = *p;
            if (pixel_value < 255/2)
            {
                // add obstacle
                b2BodyDef bd;
                bd.type = b2_staticBody;
                float x_pos = col + 0.5;
                float y_pos = map_height - row - 0.5;
                bd.position.Set(x_pos, y_pos);
                b2Body* obst = world.CreateBody(&bd);

                b2PolygonShape boxShape;
                boxShape.SetAsBox(0.5f, 0.5f);

                b2FixtureDef boxFixtureDef;
                boxFixtureDef.shape = &boxShape;
                obst->CreateFixture(&boxFixtureDef);

                obstacle_bodies.push_back(obst);
            }
            p++;  // points to each pixel value in turn assuming a CV_8UC1 greyscale image
        }
    }
}

int Environment::generate_empty_index() const
{
    // number of (flattened) indices on the grid map
    int num_index = map_image_raw.size().width * map_image_raw.size().height;
    std::vector<int> map_indices;
    map_indices.resize(num_index);
    for (int i=0; i < num_index; i++)
    {
        map_indices[i] = i;
    }

    std::random_shuffle(map_indices.begin(), map_indices.end());

    for (auto it = map_indices.begin(); it != map_indices.end(); it++)
    {
        int index = *it;

        // get position from index
        int row = index / map_image_raw.size().width;
        int col = index - row * map_image_raw.size().width;  // integer division
        float x_pos = col + 0.5;  // center of cell
        float y_pos = map_height - row - 0.5;
        b2Vec2 index_position(x_pos, y_pos);

        // check if cell has a wall
        if (static_cast<int>(map_image_raw.at<unsigned char>(row, col) < 255/2))
            continue;

        // check if any agent is near
        bool somethings_near = false;
        for (auto& agent : agent_bodies)
        {
            if ((agent->GetPosition() - index_position).Length() < robot_diam)
            {
                somethings_near = true;
                break;
            }
        }
        for (auto& goal : goal_positions)
        {
            if ((goal - index_position).Length() < robot_diam)
            {
                somethings_near = true;
                break;
            }
        }
        if (!somethings_near) return index;
    }
    // no index found, throw error
    throw std::runtime_error("No position could be generated");
}

void Environment::reset()
{
    done = false;
    episode_sim_time = 0.;

    // create new agents with new goals
    int number_of_agents_ = number_of_agents;
    while (number_of_agents > 0)
        remove_agent(number_of_agents-1);

    for (int i=0; i < number_of_agents_; i++)
        add_agent();
}


int Environment::add_agent()
{
    // create simulated body
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody;

    // generate random starting position
    int index = generate_empty_index();
    int row = index / map_image_raw.size().width;
    int col = index - row * map_image_raw.size().width;  // integer division
    float x_pos = col + 0.5;
    float y_pos = map_height - row - 0.5;
    bodyDef.position.Set(x_pos, y_pos);
    b2Body* body = world.CreateBody(&bodyDef);

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

    // generate random goal position
    index = generate_empty_index();
    row = index / map_image_raw.size().width;
    col = index - row * map_image_raw.size().width;  // integer division
    x_pos = col + 0.5;
    y_pos = map_height - row - 0.5;
    goal_positions.push_back(b2Vec2(x_pos, y_pos));

    // generate random color for agent
    int b = static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) * 255;
    int r = static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) * 255;
    int g = static_cast<float>(rand_r(&seed)) / static_cast<float>(RAND_MAX) * 255;
    cv::Scalar agent_color(b, r, g);
    agent_colors.push_back(agent_color);

    const int agent_index = number_of_agents;
    number_of_agents++;

    // fill laser_scans
    sensor_msgs::LaserScan scan;
    scan.ranges.resize(laser_nrays);
    laser_scans.push_back(scan);

    collisions.push_back(false);
    agent_lin_vel.push_back(0);
    agent_ang_vel.push_back(0);

    return agent_index;
}

void Environment::remove_agent(int agent_index)
{
    assert(agent_index >= 0 && agent_index < number_of_agents);

    world.DestroyBody(agent_bodies[agent_index]);

    agent_bodies.erase(agent_bodies.begin()+agent_index);
    goal_positions.erase(goal_positions.begin()+agent_index);
    agent_colors.erase(agent_colors.begin()+agent_index);
    number_of_agents--;
    laser_scans.erase(laser_scans.begin()+agent_index);
    collisions.erase(collisions.begin()+agent_index);
    agent_lin_vel.erase(agent_lin_vel.begin()+agent_index);
    agent_ang_vel.erase(agent_ang_vel.begin()+agent_index);
}

void Environment::step_physics()
{
    if (done == true)
        throw std::runtime_error("Attempted to step environment that is finished. Call reset() first");

    for (int j=0; j < step_multiply; j++)
    {
        for (int i=0; i < agent_bodies.size(); i++)
        {
            b2Body* agent = agent_bodies[i];
            float angle = agent->GetAngle();

            agent->SetLinearVelocity(b2Vec2(agent_lin_vel[i]*std::cos(angle), agent_lin_vel[i]*std::sin(angle)));
            agent->SetAngularVelocity(agent_ang_vel[i]);
        }

        world.Step(physics_step_size, velocity_iterations, position_iterations);
        episode_sim_time += physics_step_size;

        for (int i=0; i < agent_bodies.size(); i++)
        {
            b2Body* agent = agent_bodies[i];
            float angle = agent->GetAngle();

            // calculate laser scans
            b2Vec2 pt_from, pt_to;
            b2Vec2 position = agent->GetPosition();

            pt_from.x = robot_radius * std::cos(angle) + position.x;
            pt_from.y = robot_radius * std::sin(angle) + position.y;
            for (int j=0; j < laser_nrays; j++)
            {
                float laser_angle = angle - laser_max_angle + j * laser_max_angle * 2 / laser_nrays;
                pt_to.x = laser_max_dist * std::cos(laser_angle) + pt_from.x;
                pt_to.y = laser_max_dist * std::sin(laser_angle) + pt_from.y;

                RayCastClosestCallback callback;
                world.RayCast(&callback, pt_from, pt_to);

                float dist = laser_max_dist;
                if (callback.hit)
                    dist = (pt_from - callback.point).Length();

                laser_scans[i].ranges[j] = dist;
            }

            // check collisions
            bool hit = false;
            for (b2ContactEdge* ce = agent->GetContactList(); ce; ce = ce->next)
            {
                hit = ce->contact->IsTouching();
                if (hit)
                    break;
            }
            if (hit)
                collisions[i] = true;
        }

        // check if all the agents reached their goal
        bool check_done = true;
        for (int i=0; i < agent_bodies.size(); i++)
        {
            b2Body* agent = agent_bodies[i];
            if ((agent->GetPosition() - goal_positions[i]).Length() > robot_diam)
            {
                check_done = false;
                break;
            }
        }

        if (check_done)
        {
            done = true;
            break;
        }
    }
}

cv::Mat Environment::render(bool show, int wait)
{
    rendered_image = cv::Scalar(255, 255, 255);

    // draw obstacles
    for (int i=0; i < obstacle_bodies.size(); i++)
    {
        b2Vec2 pos = obstacle_bodies[i]->GetPosition();
        cv::Point pt1((pos.x-0.5)*scale_factor, render_height - (pos.y-0.5)*scale_factor);
        cv::Point pt2((pos.x+0.5)*scale_factor, render_height - (pos.y+0.5)*scale_factor);
        cv::rectangle(rendered_image, pt1, pt2, cv::Scalar(0, 0, 0), -1);
    }

    // draw goals
    for (int i=0; i < goal_positions.size(); i++)
    {
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
    for (int i=0; i < agent_bodies.size(); i++)
    {
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
        if (collisions[i])
        {
            // arrow
            cv::fillPoly(rendered_image, ppt, npt, 1, agent_colors[i]);

            // small circle
            cv::circle(rendered_image, center, static_cast<int>(inner_radius*scale_factor), agent_colors[i], -1);
        }
        else
        {
            // arrow
            cv::fillPoly(rendered_image, ppt, npt, 1, cv::Scalar(255, 255, 255));

            // small circle
            cv::circle(rendered_image, center, static_cast<int>(inner_radius*scale_factor),
                cv::Scalar(255, 255, 255), -1);
        }

        // number
        cv::Point textSize = cv::getTextSize(std::to_string(i), font, scale, thickness, &base_line);
        textSize.x = center.x - textSize.x / 2;
        textSize.y = center.y + textSize.y / 2;
        cv::putText(rendered_image, std::to_string(i), textSize, font, scale, cv::Scalar(0, 0, 0), thickness);

        // laser scans
        if (draw_laser)
        {
            cv::Point pt_from, pt_to;
            pt_from.x = robot_radius * scale_factor * std::cos(-angle) + center.x;
            pt_from.y = robot_radius * scale_factor * std::sin(-angle) + center.y;
            for (int j=0; j < laser_nrays; j++)
            {
                float laser_angle = angle - laser_max_angle + j * laser_max_angle * 2 / laser_nrays;
                pt_to.x = laser_scans[i].ranges[j] * scale_factor * std::cos(-laser_angle) + pt_from.x;
                pt_to.y = laser_scans[i].ranges[j] * scale_factor * std::sin(-laser_angle) + pt_from.y;
                cv::line(rendered_image, pt_from, pt_to, cv::Scalar(125, 125, 125));
            }
        }
    }

    cv::imshow("Multi-agent path finding environment", rendered_image);
    cv::waitKey(wait);

    return rendered_image;
}

EnvStep Environment::step(std::vector<Action> actions)
{
    EnvStep out;

    // process actions
    for (int i=0; i < number_of_agents; i++)
        process_action(i, actions[i]);

    // step_physics
    step_physics();

    // get observations
    for (int i=0; i < number_of_agents; i++)
        out.observations.push_back(get_observation(i));

    return out;
}

void Environment::process_action(int agent_index, Action action)
{
    agent_lin_vel[agent_index] = action.linear.x;
    agent_ang_vel[agent_index] = action.angular.z;
}

Observation Environment::get_observation(int agent_index)
{
    b2Body* agent = agent_bodies[agent_index];
    Observation obs;

    // scan
    obs.scan = laser_scans[agent_index];

    // position
    b2Vec2 position = agent_bodies[agent_index]->GetPosition();
    float yaw = agent->GetAngle();
    obs.agent_pose.x = position.x;
    obs.agent_pose.y = position.y;
    obs.agent_pose.z = yaw;

    // twist
    b2Vec2 vel = agent->GetLinearVelocity();
    float ang_vel = agent->GetAngularVelocity();
    obs.agent_twist.x = vel.x;
    obs.agent_twist.y = vel.y;
    obs.agent_twist.z = ang_vel;

    // goal pose
    obs.goal_pose.x = goal_positions[agent_index].x;
    obs.goal_pose.y = goal_positions[agent_index].y;

    // reward and done
    obs.reward = 0;
    if (done)
    {
        obs.reward = goal_reaching_reward;
    }
    else
    {
        obs.reward = step_reward;
        if (collisions[agent_index]) obs.reward += collision_reward;
    }

    collisions[agent_index] = false;

    return obs;
}

bool Environment::is_done()
{
    return done;
}

int Environment::get_number_of_agents()
{
    return number_of_agents;
}

float Environment::get_episode_sim_time()
{
    return episode_sim_time;
}

int Environment::get_observation_size()
{
    return 7 + laser_nrays;
}

std::vector<float> Environment::serialize_observation(Observation obs)
{
    std::vector<float> serialized;

    // pose
    serialized.push_back(obs.agent_pose.x);
    serialized.push_back(obs.agent_pose.y);
    serialized.push_back(obs.agent_pose.z);

    // twist
    serialized.push_back(obs.agent_twist.x);
    serialized.push_back(obs.agent_twist.z);

    // goal
    serialized.push_back(obs.goal_pose.x);
    serialized.push_back(obs.goal_pose.y);

    // scan
    serialized.insert(serialized.end(), obs.scan.ranges.begin(), obs.scan.ranges.end());

    return serialized;
}

Observation Environment::deserialize_observation(std::vector<float> obs)
{
    Observation deserialized;

    // pose
    deserialized.agent_pose.x = obs[0];
    deserialized.agent_pose.y = obs[1];
    deserialized.agent_pose.z = obs[2];

    // twist
    deserialized.agent_twist.x = obs[3];
    deserialized.agent_twist.z = obs[4];

    // goal
    deserialized.goal_pose.x = obs[5];
    deserialized.goal_pose.y = obs[6];

    // scan
    deserialized.scan.ranges.insert(deserialized.scan.ranges.begin(), obs.begin()+7, obs.end());

    return deserialized;
}
