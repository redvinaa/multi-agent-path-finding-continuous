// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_ENVIRONMENT_H
#define MAPF_ENVIRONMENT_ENVIRONMENT_H

// other headers
#include <box2d/box2d.h>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <string>
#include <random>
#include <utility>
#include <memory>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>
#include "mapf_environment/pathfinder.h"


//! \brief Type for storing agent info
using t_info = std::unordered_map<std::string, float>;


/*! \brief Environment for multi-agent path finding simulation
 *
 * This is the core (without ROS) class, that includes the physics
 * and visualization. Multiple agents can be added, and any map
 * can be used.
 */
class Environment
{
    protected:
        // box2d fields
        b2Vec2  gravity;
        b2World world;
        std::vector<b2Body*> agent_bodies;
        std::vector<b2Body*> obstacle_bodies;
        std::vector<b2Vec2> goal_positions;

        // physics properties
        float physics_step_size;
        int32 velocity_iterations, position_iterations;

        // other fields
        std::vector<cv::Scalar> agent_colors;
        std::vector<std::vector<float>> current_actions, laser_scans;
        std::vector<bool> collisions;
        std::vector<t_point> obstacle_positions;
        std::tuple<float, float> map_size;  // width, height
        std::vector<int> map_indices;
        bool draw_laser, draw_noisy_pose, draw_global_path, done;
        float block_size, scale_factor, laser_max_angle, laser_max_dist,
            robot_diam, robot_radius, goal_reaching_reward, collision_reward,
            goal_distance_reward_mult, episode_sim_time, noise, obstacle_width,
            obstacle_height, safe_pix_width, safe_pix_height;
        std::string map_path;
        int render_height, render_width, laser_nrays, step_multiply, number_of_agents,
            max_steps, current_steps, resolution_per_pix, carrot_planner_dist;
        unsigned int seed;
        cv::Mat map_image_raw, map_image, rendered_image, map_safety;
        cv::Scalar color;
        std::shared_ptr<AStar> astar;

        // random
        std::shared_ptr<std::default_random_engine> generator;
        std::normal_distribution<float> normal_dist;
        std::uniform_real_distribution<float> uniform_dist;

        /*! \brief Calculate position from pixel on a map */
        t_point pix_to_pos(const int col, const int row, const cv::Size image_size) const;

        /*! \brief Load map
         *
         * Load image at path _map_path.
         * Image coordinates use the OpenCV convention.
         */
        void init_map(void);

        /*! \brief Create box2d physics world, create boundaries and add obstacles based on the map image
         *
         * Physics coorinates use the Box2d convention.
         */
        void init_physics(void);

        /* \brief Calculates the distance between a rectangular obstacle and a point
         *
         * \param pt_x x coordinate of point
         * \param pt_y y coordinate of point
         * \param obst_x x coordinate of obstacle center
         * \param obst_y y coordinate of obstacle center
         * \return distance
         */
        float distance_from_obstacle(float pt_x, float pt_y,
            float obst_x, float obst_y) const;

        /*! \brief Randomly find an empty place on the map (no obstacles and no other agents nearby)
         *
         * \return The position of the center of the empty cell
         * \exception std::runtime_error Raised if no free cells are found
         */
        std::pair<float, float> generate_empty_position(void) const;

        /*! \brief Create an agent with physics, random starting and goal positions
         *
         * \sa generate_empty_index()
         * \return Index of the new agent (starting from 0)
         */
        void add_agent();

        /*! \brief Step physics, calculate observations (incl. scans, collisions) and rewards
         *
         * First, the stored collisions are cleared.
         * Then, the internally stored linear and angular velocities are set.
         * Then, the physics simulation step is calculated (step_multiply times).
         * Finally, the observations are updated:
         *   - laser_scans
         *   - collisions,
         * along with the other parts of the observations.
         * The last element of the observations vector is the global observation.
         *
         * Done is not set here, but in step(), which calls this function.
         *
         * After that, the environment checks if any of the agents reached their
         * goals. If yes, then for the given agents a new goal is generated.
         * \param render See step()
         * \return Tuple of the observations for the agents, infos and rewards
         * \exception std::runtime_error Raised if done == true
         * \sa get_observation(), step()
         */
        std::tuple<std::vector<std::vector<float>>, std::vector<t_info>, std::vector<float>> step_physics(bool render = false);

        /*! \brief Save the linear and angular velocity of the
         * given agent
         *
         * Action is a vector of dimension 2 (linear, angular vel).
         *
         * However, the velocities are set only in the step_physics()
         * \sa step_physics()
         */
        void process_action(int agent_index, std::vector<float> action);

        /*! \brief Calculate the observations for the given agent
         *
         * Agent -1 returns the global observations
         *
         * The agent observations contained in the vector are the following, respectively:
         *   - direction to subgoal_pose
         *   - scan, vector of ranges
         *   - other agents' rel pos
         *   - other agents' velocity
         *
         * The global observations contained in the vector are the following, respectively:
         *   - agent_poses, vector of linear x, y, and angle z
         *   - directions to subgoal_pose
         *   - distances to goal, for the absolute values
         *
         * Rewards are calculated based on the reached_goal argument.
         *
         * \sa step_physics()
         * \return Tuple of the calculated observation vector and the reward
         */
        std::tuple<std::vector<float>, float> get_observation(int agent_index, bool reached_goal);

        /*! \brief Gets the i-th point on the path of the agent
         *  as set in the variable carrot_planner_dist */
        t_point carrot_planner(const t_path route) const;

        /*! \brief Get subgoal angle and distance
         *
         *  \return Tuple of angle to subgoal, and distance
         */
        std::tuple<float, float> get_subgoal(b2Vec2 start, b2Vec2 goal) const;

        FRIEND_TEST(EnvironmentCore,    constructorRuns);
        FRIEND_TEST(EnvironmentFixture, testConstructor);
        FRIEND_TEST(EnvironmentFixture, testMap);
        FRIEND_TEST(EnvironmentFixture, testPhysics);
        FRIEND_TEST(EnvironmentFixture, testAddAndRemoveAgent);
        FRIEND_TEST(EnvironmentFixture, testReset);
        FRIEND_TEST(EnvironmentFixture, testContact);
        FRIEND_TEST(EnvironmentFixture, testMovement);
        FRIEND_TEST(EnvironmentFixture, testObservation);
        FRIEND_TEST(EnvironmentFixture, testSerialize);
        FRIEND_TEST(EnvironmentFixture, testPathFinder);
        FRIEND_TEST(CriticFixture,      testGetValues);
        FRIEND_TEST(CriticFixture,      testTraining);

    public:
        /*! \brief Sets the default parameters, and calls init_map() and init_physics()
         *
         *  One step takes 0.5 s in sim time, and one ep is 30 s.
         *
         * \param _map_path Map image to load
         * \param _map_size  Real-life size of the area, in meters, width, height
         * \param _number_of_agents Set number of agents to generate
         * \param _physics_step_size Time delta for the physics engine
         * \param _step_multiply When step_physics() gets called, step the environment this many times
         * \param _robot_diam Diameter of the simulated robots (one pixel on the map is 1 meter)
         * \param _max_steps The episode is ended after this many steps
         * \param _noise Zero mean Gaussian noise applied to agent_pose and scan in the Observations
         * \param _seed Seed to generate random numbers
         * \sa init_map(), init_physics()
         */
        Environment(
            std::string  _map_path,
            t_point      _map_size,
            int          _number_of_agents  = 2,
            unsigned int _seed              = 0,
            int          _max_steps         = 60,
            float        _robot_diam        = 0.7,
            float        _noise             = 0.00,
            float        _physics_step_size = 0.1,
            int          _step_multiply     = 5);

        /*! \brief Set done=false, generate new starting positions and goals for all agents
         * \return First observation
         */
        std::vector<std::vector<float>> reset();

        /*! \brief Draw the obstacles, agents, goals, optionally
         * the raycasts, and show collisions as well
         *
         * \return The drawn image
         */
        cv::Mat get_rendered_pic(bool debug=false);

        /*! \brief Show rendered image of the environment
         *
         * \param debug Show everything on image
         * \param wait Passed to cv::waitKey (timeout in ms)
         * \sa get_rendered_pic
         */
        void render(int wait = 0, bool debug=false);

        /* \brief Add actions, get observations, rewards and dones
         *
         * Done is set here, if max_steps is reached.
         * The last element of the observations vector is the global observation.
         * Based loosely on OpenAI Gym API
         *
         * \param actions vector of actions for every agens (size = no_agents * 2)
         * \param render If true, the env is rendered at every physics step, and waits
         * \return Tuple of vectors: (obs, reward, info, done)
         */
        std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<t_info>, std::vector<bool>>
            step(std::vector<std::vector<float>> actions, bool render = false);

        /*! \brief Is the episode over
         */
        bool is_done() const;

        /*! \brief Get number of agents
         */
        int get_number_of_agents() const;

        /*! \brief How much time has passed in the simulation
         * since the start of the episode
         *
         * Depends on physics_step_size, step_multiply and how many times
         * step_physics() was called
         *
         * \sa step_physics()
         *
         * \return Simulation time
         */
        float get_episode_sim_time() const;

        /*! \brief Return shape of observation space (nagents x obs_len)
         */
        std::vector<int> get_observation_space() const;

        /*! \brief Return shape of action space (nagents x act_len)
         */
        std::vector<int> get_action_space() const;
};

#endif  // MAPF_ENVIRONMENT_ENVIRONMENT_H
