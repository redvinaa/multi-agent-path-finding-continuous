// Copyright 2021 Reda Vince

#include "mapf_environment/pathfinder.h"
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <vector>


Node::Node(): parent(nullptr) {}

int Node::distance(const Node& other)
{
    int dist_x = std::abs(other.x - x);
    int dist_y = std::abs(other.y - y);
    if (dist_x > dist_y)
        return dist_y * 14 + (dist_x - dist_y) * 10;
    return dist_x * 14 + (dist_y - dist_x) * 10;
}


AStar::AStar(cv::Mat _map, bool _diag, bool _init_all)
{
    grid_map = _map;
    diag     = _diag;
    init_all = _init_all;

    if (init_all)
    {
        // auto time_start = std::chrono::high_resolution_clock::now();
        int n_paths = 0;

        paths.resize(grid_map.size().width, grid_map.size().height,
            grid_map.size().width, grid_map.size().height);

        for (int start_x=0; start_x < grid_map.size().width; start_x++)
        {
            for (int start_y=0; start_y < grid_map.size().height; start_y++)
            {
                if (static_cast<int>(grid_map.at<unsigned char>(start_y, start_x)) < 255/2)
                    continue;

                for (int goal_x=0; goal_x < grid_map.size().width; goal_x++)
                {
                    for (int goal_y=0; goal_y < grid_map.size().height; goal_y++)
                    {
                        if ((start_x == goal_x) and (start_y == goal_y))
                            continue;

                        if (static_cast<int>(grid_map.at<unsigned char>(goal_y, goal_x)) < 255/2)
                            continue;

                        // check if path has already been calculated in the other direction
                        t_path route;
                        float dist;
                        std::tie(route, dist) = paths(goal_x, goal_y, start_x, start_y);
                        if (route.size() > 0)
                        {
                            std::reverse(route.begin(), route.end());
                            paths(start_x, start_y, goal_x, goal_y) =
                                std::make_tuple(route, dist);
                        }
                        else
                            // calculate route
                            paths(start_x, start_y, goal_x, goal_y)
                                = find_path(start_x, start_y, goal_x, goal_y);

                        n_paths++;
                    }
                }
            }
        }
        // auto time_end  = std::chrono::high_resolution_clock::now();
        // auto time_diff = time_end - time_start;
        // float secs = static_cast<std::chrono::duration<float, std::milli>>(time_diff).count() / 1e3;
        // std::cout << "Global path planning calculated " << n_paths << " paths, in " <<
        //     secs << " s" << std::endl;
    }
}

std::tuple<t_path, float> AStar::find(int start_x, int start_y, int goal_x, int goal_y) const
{
    assert(static_cast<int>(grid_map.at<unsigned char>(start_y, start_x)) > 255/2);
    assert(static_cast<int>(grid_map.at<unsigned char>(goal_y, goal_x)) > 255/2);

    if (((start_x == goal_x) and (start_y == goal_y)))
        return std::make_tuple(t_path(), 0.);

    if (init_all)
        return paths(start_x, start_y, goal_x, goal_y);
    return find_path(start_x, start_y, goal_x, goal_y);
}

std::tuple<t_path, float> AStar::find_path(int start_x, int start_y, int goal_x, int goal_y) const
{
    assert(not (static_cast<int>(grid_map.at<unsigned char>(start_y, start_x)) < 255/2));
    assert(not (static_cast<int>(grid_map.at<unsigned char>(goal_y, goal_x)) < 255/2));
    assert(not ((start_x == goal_x) and (start_y == goal_y)));

    // create grid of nodes
    Eigen::Tensor<Node, 2> nodes;
    nodes.resize(grid_map.size().width, grid_map.size().height);

    // create opened and closed containers
    std::vector<Node*> opened, closed;

    // create start and goal node
    Node* const start = &nodes(start_x, start_y);
    Node* const goal = &nodes(goal_x, goal_y);

    start->x = start_x;
    start->y = start_y;
    start->g_cost = 0;
    start->h_cost = start->distance(*goal);
    start->f_cost = start->h_cost;
    opened.push_back(start);

    goal->x = goal_x;
    goal->y = goal_y;

    Node* current = start;

    for (int loop=0; loop < 1000; loop++)
    {
        current = opened[0];
        for (auto node : opened)
        {
            if (node->f_cost < current->f_cost or
                    (node->f_cost == current->f_cost and node->h_cost < current->h_cost))
                current = node;
        }

        opened.erase(std::find(opened.begin(), opened.end(), current));
        closed.push_back(current);

        if (current == goal)
            break;

        // foreach neighbour
        auto neighbours = get_neighbours(nodes, current);
        for (auto neig : neighbours)
        {
            if (static_cast<int>(grid_map.at<unsigned char>(neig->y, neig->x) < 255/2))
                continue;

            if (std::find(closed.begin(), closed.end(), neig) != closed.end())
                continue;

            int new_cost = current->g_cost + current->distance(*neig);
            if ((std::find(opened.begin(), opened.end(), neig) == opened.end()) or
                    new_cost < neig->g_cost)
            {
                neig->g_cost = new_cost;
                neig->h_cost = neig->distance(*goal);
                neig->f_cost = neig->g_cost + neig->h_cost;
                neig->parent = current;

                if (std::find(opened.begin(), opened.end(), neig) == opened.end())
                    opened.push_back(neig);
            }
        }
    }

    std::vector<std::tuple<int, int>> p_vec;
    current = goal;
    while (true)
    {
        auto pos = std::make_tuple(current->x, current->y);
        p_vec.push_back(pos);

        if (current->parent == nullptr)
            break;
        current = current->parent;
        if (current == start)
            break;
    }

    std::reverse(p_vec.begin(), p_vec.end());
    return std::make_tuple(p_vec, goal->f_cost / 10.);
}

std::vector<Node*> AStar::get_neighbours(Eigen::Tensor<Node, 2> &grid, const Node* node) const
{
    std::vector<Node*> neighbours;

    for (int i=-1; i < 2; i++)
    {
        for (int j=-1; j < 2; j++)
        {
            if (i == 0 and j == 0)  // center, current node
                continue;

            if (not diag and ((i == -1 and j == -1) or
                              (i == -1 and j == 1) or
                              (i == 1  and j == -1) or
                              (i == 1  and j == 1)))
                continue;

            int neig_x = node->x + i;
            int neig_y = node->y + j;
            if ((neig_x < 0) or (neig_x >= grid_map.size().width))
                continue;
            if ((neig_y < 0) or (neig_y >= grid_map.size().height))
                continue;

            Node* neig = &grid(neig_x, neig_y);
            neig->x = neig_x;
            neig->y = neig_y;

            neighbours.push_back(neig);
        }
    }

    return neighbours;
}
