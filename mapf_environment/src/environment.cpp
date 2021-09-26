// Copyright 2021 Reda Vince

#include "mapf_environment/environment.h"
#include "mapf_environment/raycast_callback.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <box2d/box2d.h>
#include <vector>
#include <tuple>
#include <string>
#include <utility>
#include <stdexcept>
#include <functional>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <random>
#include <memory>


Environment::Environment(std::string _map_path,
    int          _number_of_agents /* 2 */,
    float        _physics_step_size /* 0.01 */,
    int          _step_multiply /* 50 */,
    float        _laser_max_angle /* 45.*M_PI/180. */,
    float        _laser_max_dist /* 10. */,
    float        _robot_diam /* 0.8 */,
    int          _velocity_iterations /* 6 */,
    int          _position_iterations /* 2 */,
    int          _render_height /* 700 */,
    int          _laser_nrays /* 10 */,
    int          _max_steps /* 60 */,
    bool         _draw_laser /* false */,
    bool         _draw_noisy_pose /* false */,
    float        _goal_reaching_reward /* 1. */,
    float        _collision_reward /* -0.5 */,
    float        _noise /* 0.01 */,
    unsigned int _seed /* 0 */):
        gravity(0, 0),
        world(gravity),
        normal_dist(0., _noise),
        uniform_dist(0., 1.)
{
    map_path             = _map_path;
    number_of_agents     = _number_of_agents;
    physics_step_size    = _physics_step_size;
    step_multiply        = _step_multiply;
    laser_max_angle      = _laser_max_angle;
    laser_max_dist       = _laser_max_dist;
    robot_diam           = _robot_diam;
    velocity_iterations  = _velocity_iterations;
    position_iterations  = _position_iterations;
    render_height        = _render_height;
    laser_nrays          = _laser_nrays;
    max_steps            = _max_steps;
    draw_laser           = _draw_laser;
    draw_noisy_pose      = _draw_noisy_pose;
    goal_reaching_reward = _goal_reaching_reward;
    collision_reward     = _collision_reward;
    noise                = _noise;
    seed                 = _seed;

    generator = std::make_shared<std::default_random_engine>(_seed);

    assert(robot_diam < 1.);
    assert(number_of_agents > 0);
    assert(laser_nrays > 0);

    done = true;
    robot_radius = robot_diam/2;

    init_map();  // load map before physics
    init_physics();

    for (int i=0; i < number_of_agents; i++)
        add_agent();
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

std::pair<float, float> Environment::generate_empty_position() const
{
    // number of (flattened) indices on the grid map
    int num_index = map_image_raw.size().width * map_image_raw.size().height;
    std::vector<int> map_indices(num_index);
    for (int i=0; i < num_index; i++)
    {
        map_indices[i] = i;
    }

    std::shuffle(map_indices.begin(), map_indices.end(), *generator);

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
        if (!somethings_near)
        {
            for (auto& goal : goal_positions)
            {
                if ((goal - index_position).Length() < robot_diam)
                {
                    somethings_near = true;
                    break;
                }
            }
        }
        if (!somethings_near)
        {
            // found position, return
            return std::make_pair(x_pos, y_pos);
        }
    }
    // no index found, throw error
    throw std::runtime_error("No position could be generated");
}

std::vector<std::vector<float>> Environment::reset()
{
    if (number_of_agents < 1)
    {
        throw std::runtime_error("Add agents before resetting environment");
    }

    // create new agents with new goals
    for (int i=0; i < number_of_agents; i++)
    {
        float x_pos, y_pos;
        std::tie(x_pos, y_pos) = generate_empty_position();
        float angle = uniform_dist(*generator) * M_PI * 2;
        agent_bodies[i]->SetTransform(b2Vec2(x_pos, y_pos), angle);

        std::tie(x_pos, y_pos) = generate_empty_position();
        goal_positions[i] = b2Vec2(x_pos, y_pos);
    }

    done = false;
    episode_sim_time = 0.;
    current_steps = 0;
    std::fill(collisions.begin(), collisions.end(), false);

    // get observations
    auto obs_and_rewards = step_physics();  // no movement, but calculate laser scans
    last_observation = std::get<0>(obs_and_rewards);  // save observations for rendering

    std::vector<std::vector<float>> out(number_of_agents);
    for (int i=0; i < number_of_agents; i++)
    {
        out[i].resize(get_observation_space()[0]);
        out[i] = std::get<0>(obs_and_rewards)[i];
    }

    return out;
}

void Environment::add_agent()
{
    // create simulated body
    b2BodyDef bodyDef;
    bodyDef.type = b2_dynamicBody;

    // generate random starting position
    float x_pos, y_pos;
    std::tie(x_pos, y_pos) = generate_empty_position();
    bodyDef.position.Set(x_pos, y_pos);
    bodyDef.angle = uniform_dist(*generator) * M_PI * 2;
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
    std::tie(x_pos, y_pos) = generate_empty_position();
    goal_positions.push_back(b2Vec2(x_pos, y_pos));

    // generate random color for agent
    int b = uniform_dist(*generator) * 255;
    int r = uniform_dist(*generator) * 255;
    int g = uniform_dist(*generator) * 255;
    cv::Scalar agent_color(b, r, g);
    agent_colors.push_back(agent_color);

    // fill laser_scans
    std::vector<float> scan(laser_nrays);
    std::fill(scan.begin(), scan.end(), 0.);
    laser_scans.push_back(scan);

    collisions.push_back(false);
    current_actions.resize(number_of_agents);
    for (auto& act : current_actions)
    {
        act.push_back(0);
        act.push_back(0);
    }
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>> Environment::step_physics()
{
    if (done == true)
        throw std::runtime_error("Attempted to step environment that is finished. Call reset() first");

    std::fill(collisions.begin(), collisions.end(), false);

    std::vector<bool> reached_goal(number_of_agents);
    std::fill(reached_goal.begin(), reached_goal.end(), false);

    /* Step env step_multily number of times, and calculate collisions
     * for each step. Collisions are only set to false after the loop
     * while getting observations */
    for (int j=0; j < step_multiply; j++)
    {
        // set agent speeds
        for (int i=0; i < number_of_agents; i++)
        {
            b2Body* agent = agent_bodies[i];
            float angle = agent->GetAngle();

            agent->SetLinearVelocity(b2Vec2(current_actions[i][0]*std::cos(angle),
                current_actions[i][0]*std::sin(angle)));
            agent->SetAngularVelocity(current_actions[i][1]);
        }

        world.Step(physics_step_size, velocity_iterations, position_iterations);
        episode_sim_time += physics_step_size;

        // get collisions
        for (int i=0; i < number_of_agents; i++)
        {
            b2Body* agent = agent_bodies[i];

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

        // check if any of the agents reached their goal
        for (int i=0; i < number_of_agents; i++)
        {
            b2Body* agent = agent_bodies[i];
            bool done = ((agent->GetPosition() - goal_positions[i]).Length() < robot_diam);

            if (done)  // if done, generate new goal
            {
                float x_pos, y_pos;
                std::tie(x_pos, y_pos) = generate_empty_position();
                goal_positions[i] = b2Vec2(x_pos, y_pos);

                reached_goal[i] = true;
            }
        }
    }

    // calculate laser scans
    for (int i=0; i < number_of_agents; i++)
    {
        b2Body* agent = agent_bodies[i];
        float angle = agent->GetAngle();

        b2Vec2 pt_from, pt_to;
        b2Vec2 position = agent->GetPosition();

        pt_from.x = robot_radius * std::cos(angle) + position.x;
        pt_from.y = robot_radius * std::sin(angle) + position.y;

        // check if any of the other agents are too close,
        // otherwise RayCast does not work
        bool agent_close = false;
        for (int h=0; h < number_of_agents; h++)
        {
            if (h == i)  // same agent
                continue;

            if ((pt_from - agent_bodies[h]->GetPosition()).Length() < robot_radius)
            {
                agent_close = true;
                break;
            }
        }

        for (int j=0; j < laser_nrays; j++)
        {
            float range = laser_max_dist;

            if (agent_close)
                range = 0;
            else
            {
                float laser_angle = angle - laser_max_angle + j * laser_max_angle * 2 / laser_nrays;
                pt_to.x = laser_max_dist * std::cos(laser_angle) + pt_from.x;
                pt_to.y = laser_max_dist * std::sin(laser_angle) + pt_from.y;

                RayCastClosestCallback callback;
                world.RayCast(&callback, pt_from, pt_to);

                if (callback.hit)
                    range = (pt_from - callback.point).Length();
            }

            laser_scans[i][j] = range + normal_dist(*generator);
        }
    }

    std::tuple<std::vector<std::vector<float>>, std::vector<float>> obs_and_rewards;
    std::get<0>(obs_and_rewards).resize(number_of_agents);
    std::get<1>(obs_and_rewards).resize(number_of_agents);

    // get observations
    for (int i=0; i < number_of_agents; i++)
    {
        auto single_obs_and_reward = get_observation(i, reached_goal[i]);
        std::get<0>(obs_and_rewards)[i] = std::get<0>(single_obs_and_reward);
        std::get<1>(obs_and_rewards)[i] = std::get<1>(single_obs_and_reward);
    }

    return obs_and_rewards;
}

cv::Mat Environment::get_rendered_pic()
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
    for (int i=0; i < number_of_agents; i++)
    {
        b2Vec2 position = goal_positions[i];

        // big circle
        cv::Point center(position.x*scale_factor, render_height-position.y*scale_factor);
        cv::circle(rendered_image, center, robot_radius*scale_factor, agent_colors[i], -1);
    }

    // draw agents
    float inner_radius = robot_radius * 0.8;
    float noisy_pose_radius  = robot_radius * 0.1;
    auto font = cv::FONT_HERSHEY_TRIPLEX;
    int thickness = 3;
    int base_line = 0;
    // TODO(redvinaa): Update opencv
    // auto scale = cv::getFontScaleFromHeight(font, inner_radius*scale_factor, thickness);
    int scale = 3;
    for (int i=0; i < number_of_agents; i++)
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
            cv::fillPoly(rendered_image, ppt, npt, 1, cv::Scalar(0, 0, 0));

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

        if (draw_noisy_pose)
        {
            float noisy_pose_x = last_observation[i][0];
            float noisy_pose_y = last_observation[i][1];
            cv::Point noisy_center(noisy_pose_x*scale_factor, render_height-noisy_pose_y*scale_factor);
            cv::circle(rendered_image, noisy_center, static_cast<int>(noisy_pose_radius*scale_factor),
                cv::Scalar(0, 0, 0), 2);
            cv::circle(rendered_image, noisy_center, static_cast<int>(noisy_pose_radius*scale_factor),
                cv::Scalar(255, 255, 255), -1);
        }

        // laser scans
        if (draw_laser)
        {
            cv::Point pt_from, pt_to;
            pt_from.x = robot_radius * scale_factor * std::cos(-angle) + center.x;
            pt_from.y = robot_radius * scale_factor * std::sin(-angle) + center.y;
            for (int j=0; j < laser_nrays; j++)
            {
                float laser_angle = angle - laser_max_angle + j * laser_max_angle * 2 / laser_nrays;
                pt_to.x = laser_scans[i][j] * scale_factor * std::cos(-laser_angle) + pt_from.x;
                pt_to.y = laser_scans[i][j] * scale_factor * std::sin(-laser_angle) + pt_from.y;
                cv::line(rendered_image, pt_from, pt_to, cv::Scalar(125, 125, 125));
            }
        }
    }

    return rendered_image;
}

void Environment::render(int wait)
{
    cv::imshow("Multi-agent path finding environment", get_rendered_pic());
    cv::waitKey(wait);
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>>
    Environment::step(std::vector<std::vector<float>> actions)
{
    assert(actions.size() == number_of_agents);
    for (auto act : actions)
    {
        assert(act.size() == 2);
        assert(act[0] >= 0.);
    }

    // process actions
    for (int i=0; i < number_of_agents; i++)
    {
        process_action(i, actions[i]);
    }

    std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>> ret;
    std::get<0>(ret).resize(number_of_agents);
    std::get<1>(ret).resize(number_of_agents);
    std::get<2>(ret).resize(number_of_agents);

    auto obs_and_reward = step_physics();
    last_observation = std::get<0>(obs_and_reward);  // save observations for rendering
    current_steps++;

    for (int i=0; i < number_of_agents; i++)
    {
        std::get<0>(ret)[i] = std::get<0>(obs_and_reward)[i];
        std::get<1>(ret)[i] = std::get<1>(obs_and_reward)[i];
    }

    // set done
    if (current_steps >= max_steps)
        done = true;

    std::fill(std::get<2>(ret).begin(), std::get<2>(ret).end(), done);

    return ret;
}

void Environment::process_action(int agent_index, std::vector<float> action)
{
    assert(action.size() == 2);
    assert(std::abs(action[0]) < 1.     + 1e-3);  // max 1 m/s
    assert(std::abs(action[1]) < M_PI/2 + 1e-3);  // max 90 deg/s

    current_actions[agent_index] = action;
}

std::tuple<std::vector<float>, float> Environment::get_observation(int agent_index, bool reached_goal)
{
    b2Body* agent = agent_bodies[agent_index];
    std::tuple<std::vector<float>, float> obs_and_reward;

    // position
    b2Vec2 position = agent_bodies[agent_index]->GetPosition();

    float yaw = agent->GetAngle();
    yaw = std::max(yaw, static_cast<float>(0.));  // normalize angle between 0 and pi
    yaw = std::min(yaw, static_cast<float>(M_PI));

    std::get<0>(obs_and_reward).push_back(position.x + normal_dist(*generator));
    std::get<0>(obs_and_reward).push_back(position.y + normal_dist(*generator));
    std::get<0>(obs_and_reward).push_back(yaw        + normal_dist(*generator));

    // twist
    b2Vec2 vel = agent->GetLinearVelocity();
    float lin_vel = std::sqrt(std::pow(vel.x, 2) + std::pow(vel.y, 2));
    float ang_vel = agent->GetAngularVelocity();
    std::get<0>(obs_and_reward).push_back(lin_vel + normal_dist(*generator));
    std::get<0>(obs_and_reward).push_back(ang_vel + normal_dist(*generator));

    // goal pose
    std::get<0>(obs_and_reward).push_back(goal_positions[agent_index].x);
    std::get<0>(obs_and_reward).push_back(goal_positions[agent_index].y);

    // scan
    std::get<0>(obs_and_reward).insert(std::get<0>(obs_and_reward).end(),
        laser_scans[agent_index].begin(), laser_scans[agent_index].end());

    // reward and done
    float reward = 0.;
    if (reached_goal)
        reward = goal_reaching_reward;
    else if (collisions[agent_index])
        reward = collision_reward;
    std::get<1>(obs_and_reward) = reward;

    return obs_and_reward;
}

bool Environment::is_done() const
{
    return done;
}

int Environment::get_number_of_agents() const
{
    return number_of_agents;
}

float Environment::get_episode_sim_time() const
{
    return episode_sim_time;
}

std::vector<int> Environment::get_observation_space() const
{
    int single_obs_len = 7 + laser_nrays;
    std::vector<int> ret(number_of_agents);
    std::fill(ret.begin(), ret.end(), single_obs_len);
    return ret;
}

std::vector<int> Environment::get_action_space() const
{
    int single_act_len = 2;
    std::vector<int> ret(number_of_agents);
    std::fill(ret.begin(), ret.end(), single_act_len);
    return ret;
}
