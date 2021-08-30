// Copyright 2021 Reda Vince

#ifndef MAPF_ENVIRONMENT_ENVIRONMENT_H
#define MAPF_ENVIRONMENT_ENVIRONMENT_H

// ros datatype headers
#include "mapf_environment/types.h"

// other headers
#include <box2d/box2d.h>
#include <vector>
#include <string>
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
    private:
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
        std::vector<float> agent_lin_vel, agent_ang_vel;
        std::vector<bool> collisions;
        std::vector<LaserScan> laser_scans;
        bool draw_laser, done;
        float block_size, scale_factor, laser_max_angle, laser_max_dist,
            robot_diam, robot_radius, map_width, map_height,
            goal_reaching_reward, collision_reward, step_reward, episode_sim_time;
        std::string map_path;
        int render_height, laser_nrays, step_multiply, number_of_agents;
        unsigned int seed;
        cv::Mat map_image_raw, map_image, rendered_image;
        cv::Scalar color;

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
         * \return The flattened (row*image_width + col) index of the cell that is found to be free.
         * \exception std::runtime_error Raised if no free cells are found
         */
        int generate_empty_index(void) const;

        /*! \brief Step physics, calculate scans, collisions
         *
         * First the internally stored linear and angular velocities are set.
         * Then, the physics simulation step is calculated.
         * Finally, the observations are updated:
         *   - laser_scans
         *   - collisions
         *
         * After that, the environment checks if all the agents reached their
         * goals. If yes, then done=true is set.
         * However, collisions are not set to false here, so as not to miss
         * any. It is set to false after an observation is queried for the agent.
         * \return Index of the new agent (starting from 0)
         * \exception std::runtime_error Raised if done=true
         * \sa get_observation()
         */
        void step_physics();

        /*! \brief Save the linear and angular velocity of the
         * given agent
         *
         * However, the velocities are set only in the step_physics()
         * \sa step_physics()
         */
        void process_action(int agent_index, Action action);

        /*! \brief Calculate the observations for the given agent
         *
         * The observations are the following:
         *   - sensor_msgs/LaserScan scan
         *   - geometry_msgs/Point agent_pose  # linear x, y, and angle z
         *   - geometry_msgs/Point agent_twist # linear x, and angle z
         *   - geometry_msgs/Point goal_pose # x, y only
         *   - float32 reward
         *   - bool done
         *
         * After the observation for the given agent is queried,
         * the collision for that agent is set to false,
         * so it is important that this function is called at most once
         * per physics step. Also, the collisions are only set false here,
         * but not in step_physics()
         *
         * \sa step_physics()
         * \return The calculated observation
         */
        Observation get_observation(int agent_index);

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
         * \param _physics_step_size Time delta for the physics engine
         * \param _step_multiply When step_physics() gets called, step the environment this many times
         * \param _laser_max_angle The angle of the laser raycast from the centerline
         * \param _laser_max_dist Maximum distance for the raycast
         * \param _robot_diam Diameter of the simulated robots (one pixel on the map is 1 meter)
         * \param _velocity_iterations Parameter of the box2d physics engine
         * \param _position_iterations Parameter of the box2d physics engine
         * \param _render_height Height of the rendered image (from which the width is given)
         * \param _laser_nrays Number of rays for the raycast
         * \param _draw_laser Wether to show the raycasts on the rendered image
         * \param _goal_reaching_reward Reward for reaching the goal (only if all the other agents reach their goal too)
         * \param _collision_reward Added reward in the case of a collision
         * \param _step_reward Reward added in every step, except when the goal is reached
         * \param _seed Seed to generate random numbers
         * \sa init_map(), init_physics()
         */
        Environment(std::string _map_path,
            float        _physics_step_size    = 0.01,
            int          _step_multiply        = 5,
            float        _laser_max_angle      = 45.*M_PI/180.,
            float        _laser_max_dist       = 10.,
            float        _robot_diam           = 0.8,
            int          _velocity_iterations  = 6,
            int          _position_iterations  = 2,
            int          _render_height        = 700,
            int          _laser_nrays          = 10,
            bool         _draw_laser           = false,
            float        _goal_reaching_reward = 0.,
            float        _collision_reward     = -1.,
            float        _step_reward          = -1.,
            unsigned int _seed                 = 0);

        /*! \brief Set done=false, generate new starting positions and goals for all agents
         */
        void reset();

        /*! \brief Create an agent with physics, random starting and goal positions
         *
         * \sa generate_empty_index()
         * \return Index of the new agent (starting from 0)
         */
        int add_agent();

        /*! \brief Remove the given agent and its goal from the simulation
         *
         * \sa generate_empty_index()
         * \return Index of the new agent (starting from 0)
         */
        void remove_agent(int agent_index);

         * the raycasts, and show collisions as well
         *
         *
         */

        /* \brief Add actions, get observations, rewards and done
         *
         * Based on OpenAI Gym API
         */
        EnvStep step(CollectiveAction actions);

        /*! \brief Is the episode over
         */
        bool is_done();

        /*! \brief Get number of agents
         */
        int get_number_of_agents();

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
        float get_episode_sim_time();

        /*! \brief Calculate how many relevant elements an observation has
         */
        int get_observation_size();

        /*! \brief Take the Observation structure and
         * put the relevant data in a float vector (STRIPS REWARD)
         *
         * The reward is not serialized, because the reward is not going
         * to be fed to an ANN, which is the purpose of this function.
         *
         * \return Serialized (vectorized) observation
         *
         * \sa deserialize_observation()
         */
        static std::vector<float> serialize_observation(Observation obs);

        /*! \brief Take the observation data as a float vector
         * and construct an Observation object out of it (REWARD IS EMPTY)
         *
         * Because the serialized observation does not contain the reward,
         * that of the deserialized one is set to zero.
         *
         * \return Deserialized (Observation type) observation
         *
         * \sa serialize_observation()
         */
        static Observation deserialize_observation(std::vector<float> obs);
};

#endif  // MAPF_ENVIRONMENT_ENVIRONMENT_H
