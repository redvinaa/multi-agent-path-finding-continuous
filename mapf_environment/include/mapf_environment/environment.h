// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_ENVIRONMENT_H
#define MAPF_ENVIRONMENT_ENVIRONMENT_H

// other headers
#include <box2d/box2d.h>
#include <vector>
#include <tuple>
#include <string>
#include <random>
#include <utility>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <gtest/gtest.h>


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
        double physics_step_size;
        int32 velocity_iterations, position_iterations;

        // other fields
        std::vector<cv::Scalar> agent_colors;
        std::vector<std::vector<float>> current_actions;
        std::vector<bool> collisions;
        std::vector<std::vector<float>> laser_scans;
        std::vector<std::vector<float>> last_observation;
        bool draw_laser, draw_noisy_pose, done;
        float block_size, scale_factor, laser_max_angle, laser_max_dist,
            robot_diam, robot_radius, map_width, map_height,
            goal_reaching_reward, collision_reward, step_reward,
            episode_sim_time, noise;
        std::string map_path;
        int render_height, laser_nrays, step_multiply,
            number_of_agents, max_steps, current_steps;
        unsigned int seed;
        cv::Mat map_image_raw, map_image, rendered_image;
        cv::Scalar color;

        std::default_random_engine generator;
        std::normal_distribution<float> dist;

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
         *
         * Then, the internally stored linear and angular velocities are set.
         * Then, the physics simulation step is calculated (step_multiply times).
         * Finally, the observations are updated:
         *   - laser_scans
         *   - collisions,
         * along with the other parts of the observations.
         *
         * Done is not set here, but in step(), which calls this function.
         *
         * After that, the environment checks if any of the agents reached their
         * goals. If yes, then for the given agents a new goal is generated.
         * \return Tuple of the observations for the agents and rewards
         * \exception std::runtime_error Raised if done == true
         * \sa get_observation(), step()
         */
        std::tuple<std::vector<std::vector<float>>, std::vector<float>> step_physics();

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
         * The observations contained in the vector are the following, respectively:
         *   - agent_pose:  linear x, y, and angle z
         *   - agent_twist: linear x, and angular z
         *   - goal_pose:   linear x, y only
         *   - scan:        vector of ranges
         *
         * Rewards are calculated based on the reached_goal argument.
         *
         * \sa step_physics()
         * \return Tuple of the calculated observation vector and the reward
         */
        std::tuple<std::vector<float>, float> get_observation(int agent_index, bool reached_goal);

        FRIEND_TEST(EnvironmentCore, constructorRuns);
        FRIEND_TEST(EnvironmentFixture, testConstructor);
        FRIEND_TEST(EnvironmentFixture, testMap);
        FRIEND_TEST(EnvironmentFixture, testPhysics);
        FRIEND_TEST(EnvironmentFixture, testAddAndRemoveAgent);
        FRIEND_TEST(EnvironmentFixture, testReset);
        FRIEND_TEST(EnvironmentFixture, testContact);
        FRIEND_TEST(EnvironmentFixture, testMovement);
        FRIEND_TEST(EnvironmentFixture, testObservation);
        FRIEND_TEST(EnvironmentFixture, testSerialize);
        FRIEND_TEST(CriticFixture,      testGetValues);
        FRIEND_TEST(CriticFixture,      testTraining);

    public:
        /*! \brief Sets the default parameters, and calls init_map() and init_physics()
         *
         * \param _map_path Map image to load
         * \param _number_of_agents Set number of agents to generate
         * \param _physics_step_size Time delta for the physics engine
         * \param _step_multiply When step_physics() gets called, step the environment this many times
         * \param _laser_max_angle The angle of the laser raycast from the centerline
         * \param _laser_max_dist Maximum distance for the raycast
         * \param _robot_diam Diameter of the simulated robots (one pixel on the map is 1 meter)
         * \param _velocity_iterations Parameter of the box2d physics engine
         * \param _position_iterations Parameter of the box2d physics engine
         * \param _render_height Height of the rendered image (from which the width is given)
         * \param _laser_nrays Number of rays for the raycast
         * \param _max_steps The episode is ended after this many steps
         * \param _draw_laser Wether to show the raycasts on the rendered image
         * \param _draw_noisy_pose Wether to show on rendering the agents' position with noise
         * \param _goal_reaching_reward Reward for reaching the goal (only if all the other agents reach their goal too)
         * \param _collision_reward Added reward in the case of a collision
         * \param _step_reward Reward added in every step, except when the goal is reached
         * \param _noise Zero mean Gaussian noise applied to agent_pose and scan in the Observations
         * \param _seed Seed to generate random numbers
         * \sa init_map(), init_physics()
         */
        Environment(std::string _map_path,
            int          _number_of_agents     = 2,
            float        _physics_step_size    = 0.01,
            int          _step_multiply        = 50,
            float        _laser_max_angle      = 45.*M_PI/180.,
            float        _laser_max_dist       = 10.,
            float        _robot_diam           = 0.8,
            int          _velocity_iterations  = 6,
            int          _position_iterations  = 2,
            int          _render_height        = 700,
            int          _laser_nrays          = 10,
            int          _max_steps            = 60,
            bool         _draw_laser           = false,
            bool         _draw_noisy_pose      = false,
            float        _goal_reaching_reward = 1.,
            float        _collision_reward     = -0.5,
            float        _step_reward          = -0.1,
            float        _noise                = 0.01,
            unsigned int _seed                 = 0);

        /*! \brief Set done=false, generate new starting positions and goals for all agents
         * \return First observation
         */
        std::vector<std::vector<float>> reset();

        /*! \brief Draw the obstacles, agents, goals, optionally
         * the raycasts, and show collisions as well
         *
         * \return The drawn image
         */
        cv::Mat get_rendered_pic();

        /*! \brief Show rendered image of the environment
         *
         * \param wait Passed to cv::waitKey (timeout in ms)
         * \sa get_rendered_pic
         */
        void render(int wait = 0);

        /* \brief Add actions, get observations, rewards and dones
         *
         * Done is set here, if max_steps is reached.
         * Based loosely on OpenAI Gym API
         *
         * \param actions vector of actions for every agens (size = no_agents * 2)
         * \return Tuple of vectors: (obs, reward, done)
         */
        std::tuple<std::vector<std::vector<float>>, std::vector<float>, std::vector<bool>>
            step(std::vector<std::vector<float>> actions);

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
