// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_PATHFINDER_H
#define MAPF_ENVIRONMENT_PATHFINDER_H

#include <opencv2/opencv.hpp>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>


/*! \brief 2d tensor of path indices */
using t_path = std::vector<std::tuple<int, int>>;

struct Node
{
    int x;
    int y;

    Node* parent;

    int g_cost;
    int h_cost;
    int f_cost;

    int distance(const Node& other);

    Node();
};


/*! \brief A* path finding class
 *
 * For global path planning.
 * Based on https://github.com/quantumelixir/pathfinding.git
 */
class AStar
{
    private:
        cv::Mat grid_map;
        bool diag, init_all;
        Eigen::Tensor<t_path, 4> paths;

        /*! \brief This implements the A* algorithm */
        t_path find_path(int start_x, int start_y, int goal_x, int goal_y) const;

        /*! \brief Calculates the neighbours of the node
         *
         *  This also sets x and y of a node if it is not yet initialized.
         *
         *  \param grid Grid of nodes on which to search and init
         *  \param node Pointer to node that's neighbours are needed
         *  \return Vector of pointers to nodes
         */
        std::vector<Node*> get_neighbours(Eigen::Tensor<Node, 2> &grid, const Node* node) const;

    public:
        /*! \param _map Image to plan on
         *  \param _diag Search diagonally
         *  \param _init_all Calculate all the paths beforehand
         */
        AStar(cv::Mat _map, bool _diag, bool _init_all);

        /*! \brief Return internally stored path for the given endpoints
         *
         *  \return Vector of indicies that form the path
         */
        t_path find(int start_x, int start_y, int goal_x, int goal_y) const;
};

#endif  // MAPF_ENVIRONMENT_PATHFINDER_H
