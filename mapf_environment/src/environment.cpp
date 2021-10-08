// Copyright 2021 Reda Vince

#include "mapf_environment/environment.h"
#include "mapf_environment/raycast_callback.h"
#include "mapf_environment/pathfinder.h"
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


Environment::Environment(
    std::string  _map_path,
    t_point      _map_size,
    int          _number_of_agents /* 2 */,
    unsigned int _seed /* 0 */,
    int          _max_steps /* 30 */,
    float        _robot_diam /* 0.8 */,
    float        _noise /* 0.00 */,
    float        _physics_step_size /* 0.1 */,
    int          _step_multiply /* 10 */):
        gravity(0, 0),
        world(gravity),
        normal_dist(0., _noise),
        uniform_dist(0., 1.)
{
    map_path                  = _map_path;
    map_size                  = _map_size;
    number_of_agents          = _number_of_agents;
    seed                      = _seed;
    max_steps                 = _max_steps;
    robot_diam                = _robot_diam;
    noise                     = _noise;
    physics_step_size         = _physics_step_size;
    step_multiply             = _step_multiply;

    // hardcoded params
    laser_max_angle           = 45. * M_PI / 180.;
    laser_max_dist            = 10.;
    velocity_iterations       = 6;
    position_iterations       = 2;
    render_height             = 900;
    laser_nrays               = 10;
    draw_laser                = false;
    draw_noisy_pose           = false;
    draw_global_path          = true;
    goal_reaching_reward      = 1.;
    collision_reward          = -0.5;
    goal_distance_reward_mult = -0.05;
    resolution_per_pix        = 5;
    carrot_planner_dist       = 5;

    generator = std::make_shared<std::default_random_engine>(_seed);

    assert(number_of_agents > 0);
    assert(laser_nrays > 0);

    done = true;
    robot_radius = robot_diam/2;

    init_map();  // load map before physics
    init_physics();

    // calculate paths for map_safety
    // also where the walls are too close (grey area)
    cv::Mat map_safety_ext;
    map_image_raw.copyTo(map_safety_ext);
    cv::resize(map_safety_ext, map_safety_ext, map_safety.size(), 0, 0, cv::INTER_AREA);
    astar = std::make_shared<AStar>(map_safety_ext, true, true);

    for (int i=0; i < number_of_agents; i++)
        add_agent();
}

void Environment::init_map(void)
{
    map_image_raw = cv::imread(map_path, cv::IMREAD_GRAYSCALE);
    if (map_image_raw.empty()) throw std::runtime_error("Map image not found: '" + map_path + "'");

    safe_pix_width  = std::get<0>(map_size) / map_image_raw.size().width / resolution_per_pix;
    safe_pix_height = std::get<1>(map_size) / map_image_raw.size().height / resolution_per_pix;

    render_width = render_height * map_image_raw.size().width / map_image_raw.size().height;

    // size in meters to size in rendered pixels
    scale_factor = static_cast<float>(render_height) / std::get<1>(map_size);

    cv::resize(map_image_raw, map_image, cv::Size(render_width, render_height), 0, 0, cv::INTER_AREA);
    cv::cvtColor(map_image, map_image, cv::COLOR_GRAY2BGR);
    map_image.copyTo(rendered_image);  // set size of rendered image

    obstacle_width  = std::get<0>(map_size) / map_image_raw.size().width;
    obstacle_height = std::get<1>(map_size) / map_image_raw.size().height;

    // calculate obstacle positions
    for (int row = 0; row < map_image_raw.rows; ++row)
    {
        uchar* p = map_image_raw.ptr(row);
        for (int col = 0; col < map_image_raw.cols; ++col)
        {
            int pixel_value = *p;
            if (pixel_value < 255/2)  // 0 means obstacle
            {
                auto pos = pix_to_pos(col, row, map_image_raw.size());
                obstacle_positions.push_back(pos);
            }
            p++;  // points to each pixel value in turn assuming a CV_8UC1 greyscale image
        }
    }

    // create map with safety zones
    // used for spawning agents, goals and global path planning
    float safety_dist = robot_radius + 1e-3;

    map_image_raw.copyTo(map_safety);
    cv::resize(map_safety, map_safety, cv::Size(map_image_raw.size().width * resolution_per_pix,
        map_image_raw.size().height * resolution_per_pix), 0, 0, cv::INTER_AREA);

    for (int row = 0; row < map_safety.rows; ++row)
    {
        uchar* p = map_safety.ptr(row);
        for (int col = 0; col < map_safety.cols; ++col)
        {
            int pixel_value = *p;
            if (pixel_value > 255./2)  // no obstacle
            {
                float x_pos, y_pos;
                std::tie(x_pos, y_pos) = pix_to_pos(col, row, map_safety.size());

                if ((x_pos < safety_dist) || ((std::get<0>(map_size) - x_pos) < safety_dist))
                    *p = 50;
                else if ((y_pos < safety_dist) || ((std::get<1>(map_size) - y_pos) < safety_dist))
                    *p = 50;
                else
                    for (auto obst_pos : obstacle_positions)
                    {
                        float dist = distance_from_obstacle(x_pos, y_pos,
                            std::get<0>(obst_pos), std::get<1>(obst_pos));

                        if (dist < safety_dist)  // too close
                        {
                            *p = 50;
                            break;
                        }
                    }
            }
            p++;
        }
    }

    // create vector that stores the indices of free cells
    int num_index = map_safety.size().width * map_safety.size().height;
    map_indices.reserve(num_index);

    for (int i=0; i < num_index; i++)
    {
        int col = i % map_safety.size().width;
        int row = (i - col) / map_safety.size().width;

        if (static_cast<int>(map_safety.at<unsigned char>(row, col)) > 255/2)
            map_indices.push_back(i);
    }
    map_indices.shrink_to_fit();
}

t_point Environment::pix_to_pos(const int col, const int row, const cv::Size image_size) const
{
    float x_pos = (col + 0.5) * std::get<0>(map_size) / static_cast<float>(image_size.width);
    float y_pos = std::get<1>(map_size) - (row + 0.5) * std::get<1>(map_size) / static_cast<float>(image_size.height);
    return std::make_tuple(x_pos, y_pos);
}

void Environment::init_physics(void)
{
    float map_width  = std::get<0>(map_size);
    float map_height = std::get<1>(map_size);

    // create bounding box, (0, 0) is the bottom left corner of the map
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
    for (auto obst_pos : obstacle_positions)
    {
            // add obstacle
            b2BodyDef bd;
            bd.type = b2_staticBody;
            bd.position.Set(std::get<0>(obst_pos), std::get<1>(obst_pos));
            b2Body* obst = world.CreateBody(&bd);

            b2PolygonShape boxShape;
            boxShape.SetAsBox(obstacle_width/2, obstacle_height/2);

            b2FixtureDef boxFixtureDef;
            boxFixtureDef.shape = &boxShape;
            obst->CreateFixture(&boxFixtureDef);

            obstacle_bodies.push_back(obst);
    }
}

float Environment::distance_from_obstacle(float pt_x, float pt_y, float obst_x, float obst_y) const
{
    float obst_bot   = obst_y - obstacle_height / 2.;
    float obst_top   = obst_y + obstacle_height / 2.;
    float obst_left  = obst_x - obstacle_width  / 2.;
    float obst_right = obst_x + obstacle_width  / 2.;

    std::vector<float> x_vals = {obst_left - pt_x, 0, pt_x - obst_right};
    std::vector<float> y_vals = {obst_bot  - pt_y, 0, pt_y - obst_top};
    float dx = *std::max_element(x_vals.begin(), x_vals.end());
    float dy = *std::max_element(y_vals.begin(), y_vals.end());
    return std::sqrt(dx*dx + dy*dy);
}

std::pair<float, float> Environment::generate_empty_position() const
{

    auto map_indices_cpy = map_indices;
    std::shuffle(map_indices_cpy.begin(), map_indices_cpy.end(), *generator);

    for (auto it = map_indices_cpy.begin(); it != map_indices_cpy.end(); it++)
    {
        int index = *it;

        int col = index % map_safety.size().width;
        int row = (index - col) / map_safety.size().width;  // integer division
        float x_pos, y_pos;
        std::tie(x_pos, y_pos) = pix_to_pos(col, row, map_safety.size());
        b2Vec2 index_position(x_pos, y_pos);

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

std::tuple<std::vector<std::vector<float>>, std::vector<float>> Environment::step_physics(bool render/*=false*/)
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

        if (render)
        {
            this->render(physics_step_size * 1000);
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

cv::Mat Environment::get_rendered_pic(bool debug/*=false*/)
{
    assert(not done);
    if (debug)
    {
        map_safety.copyTo(rendered_image);
        cv::cvtColor(map_safety, rendered_image, cv::COLOR_GRAY2BGR);
        cv::resize(rendered_image, rendered_image, cv::Size(render_width, render_height), 0, 0, cv::INTER_AREA);
    }
    else
        rendered_image = cv::Scalar(255, 255, 255);

    // draw obstacles
    for (auto obst : obstacle_positions)
    {
        int pos_left   = (std::get<0>(obst) - obstacle_width/2.) * scale_factor;
        int pos_right  = (std::get<0>(obst) + obstacle_width/2.) * scale_factor;
        int pos_top    = render_height - (std::get<1>(obst) - obstacle_height/2.) * scale_factor;
        int pos_bottom = render_height - (std::get<1>(obst) + obstacle_height/2.) * scale_factor;

        cv::Point pt1(pos_left, pos_bottom);
        cv::Point pt2(pos_right, pos_top);
        cv::rectangle(rendered_image, pt1, pt2, cv::Scalar(0, 0, 0), -1);
    }

    // draw paths
    if (draw_global_path or debug)
    {
        for (int i=0; i < number_of_agents; i++)
        {
            b2Vec2 agent_pos = agent_bodies[i]->GetPosition();
            b2Vec2 goal_pos  = goal_positions[i];

            // transform coordinates to pixels
            int start_x = agent_pos.x / safe_pix_width;
            int start_y = map_safety.size().height - agent_pos.y / safe_pix_height;
            int goal_x  = goal_pos.x / safe_pix_width;
            int goal_y  = map_safety.size().height - goal_pos.y / safe_pix_height;

            t_path route = std::get<0>(astar->find(start_x, start_y, goal_x, goal_y));

            // draw point from carrot_planner
            t_point carrot_pt = carrot_planner(route);
            int pt_x = (std::get<0>(carrot_pt)+0.5) * render_width / map_safety.size().width;
            int pt_y = (std::get<1>(carrot_pt)+0.5) * render_height / map_safety.size().height;
            cv::Point carrot_pt_cv(pt_x, pt_y);
            cv::circle(rendered_image, carrot_pt_cv, robot_radius*scale_factor/5, agent_colors[i], -1);

            // transform path to cv array
            int n_pts = route.size() + 1;
            cv::Point points[n_pts];
            points[0] = cv::Point(
                (start_x+0.5) * render_width / map_safety.size().width,
                (start_y+0.5) * render_height / map_safety.size().height);
            for (int i=1; i < n_pts; i++)
            {
                int pt_x = (std::get<0>(route[i-1])+0.5) * render_width / map_safety.size().width;
                int pt_y = (std::get<1>(route[i-1])+0.5) * render_height / map_safety.size().height;
                points[i] = cv::Point(pt_x, pt_y);
            }
            const cv::Point* ps = points;
            cv::polylines(rendered_image, &ps, &n_pts, 1, /*closed=*/false, agent_colors[i], 2);
        }
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

        if (draw_noisy_pose or debug)
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
        if (draw_laser or debug)
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

void Environment::render(int wait, bool debug/*=false*/)
{
    cv::imshow("Multi-agent path finding environment", get_rendered_pic(debug));
    cv::waitKey(wait);
}

t_point Environment::carrot_planner(const t_path route) const
{
    int idx = std::min<float>(route.size(), carrot_planner_dist);
    return route[idx-1];
}

std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>>
    Environment::step(std::vector<std::vector<float>> actions, bool render/*=false*/)
{
    assert(actions.size() == number_of_agents);
    for (auto act : actions)
        assert(act.size() == 2);

    // process actions
    for (int i=0; i < number_of_agents; i++)
        process_action(i, actions[i]);

    std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>> ret;
    std::get<0>(ret).resize(number_of_agents);
    std::get<1>(ret).resize(number_of_agents);
    std::get<2>(ret).resize(number_of_agents);

    auto obs_and_reward = step_physics(render);
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
    // speeds are not meant to be restricted here,
    // this is just for making sure that the
    // input values are realistic
    assert(std::abs(action[0]) < 10.     + 1e-3);  // max 10 m/s
    assert(std::abs(action[1]) < M_PI*2  + 1e-3);  // max 360 deg/s

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

    // subgoal pose and distance
    b2Vec2 agent_pos = agent_bodies[agent_index]->GetPosition();
    b2Vec2 goal_pos  = goal_positions[agent_index];

    int start_x = agent_pos.x / safe_pix_width;
    int start_y = map_safety.size().height - agent_pos.y / safe_pix_height;
    int goal_x  = goal_pos.x / safe_pix_width;
    int goal_y  = map_safety.size().height - goal_pos.y / safe_pix_height;

    t_path route;
    float dist;
    std::tie(route, dist) = astar->find(start_x, start_y, goal_x, goal_y);

    float carr_x, carr_y;
    std::tie(carr_x, carr_y) = carrot_planner(route);

    carr_x = (carr_x + 0.5) * safe_pix_width - agent_pos.x;
    carr_y = std::get<1>(map_size) - (carr_y + 0.5) * safe_pix_height - agent_pos.y;

    std::get<0>(obs_and_reward).push_back(carr_x);
    std::get<0>(obs_and_reward).push_back(carr_y);
    std::get<0>(obs_and_reward).push_back(dist);

    // scan
    std::get<0>(obs_and_reward).insert(std::get<0>(obs_and_reward).end(),
        laser_scans[agent_index].begin(), laser_scans[agent_index].end());

    // reward and done
    float reward = 0.;
    if (reached_goal)
        reward = goal_reaching_reward;
    else if (collisions[agent_index])
        reward = collision_reward;

    reward += std::sqrt(dist) * goal_distance_reward_mult;

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
    int single_obs_len = 8 + laser_nrays;
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
