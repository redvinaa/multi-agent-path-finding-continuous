// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_PATHFINDER_H
#define MAPF_ENVIRONMENT_PATHFINDER_H

#include <opencv2/opencv.hpp>
#include <Eigen/CXX11/Tensor>


/*! \brief 2d tensor of path indices */
using t_path = std::vector<std::tuple<int, int>>;

/*! \brief x, y coordinates */
using t_point  = std::tuple<float, float>;

/*! \brief Data structure used by the A* algorithm */
struct Node
{
    int x;
    int y;

    Node* parent;

    /*! \brief Distance from start */
    int g_cost;

    /*! \brief Distance to goal */
    int h_cost;

    /*! \brief Sum of g_cost and h_cost */
    int f_cost;

    /*! \brief Finds distance between this and another node,
     *  not taking obstacles into consideration */
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
        Eigen::Tensor<std::tuple<t_path, float>, 4> paths;

        /*! \brief This implements the A* algorithm */
        std::tuple<t_path, float> find_path(int start_x, int start_y, int goal_x, int goal_y) const;

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
         *  \return Vector of indicies that form the path, and approx. distance to goal
         */
        std::tuple<t_path, float> find(int start_x, int start_y, int goal_x, int goal_y) const;
};

#endif  // MAPF_ENVIRONMENT_PATHFINDER_H
