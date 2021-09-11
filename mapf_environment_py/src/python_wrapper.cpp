// Copyright 2021 Reda Vince

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mapf_environment/environment.h"
#include <string>

namespace py = pybind11;

PYBIND11_MODULE(mapf_env, m)
{
    m.doc() = "Multi-agent path finding environment python wrapper";

    py::class_<Environment>(m, "Environment", "Environment for multi-agent path finding simulation")
    .def(py::init<std::string, int, float, int, float, float, float, int, int, int, int, int,
        bool, bool, float, float, float, unsigned int>(),
        py::arg("map_path"),
        py::arg("number_of_agents")     = 2,
        py::arg("physics_step_size")    = 0.01,
        py::arg("step_multiply")        = 50,
        py::arg("laser_max_angle")      = 45.*M_PI/180.,
        py::arg("laser_max_dist")       = 10.,
        py::arg("robot_diam")           = 0.8,
        py::arg("velocity_iterations")  = 6,
        py::arg("position_iterations")  = 2,
        py::arg("render_height")        = 700,
        py::arg("laser_nrays")          = 10,
        py::arg("max_steps")            = 60,
        py::arg("draw_laser")           = false,
        py::arg("draw_noisy_pose")      = false,
        py::arg("goal_reaching_reward") = 1.,
        py::arg("collision_reward")     = -0.5,
        py::arg("noise")                = 0.01,
        py::arg("seed")                 = 0
        )
    .def("reset",        &Environment::reset,
        "Set done=false, generate new starting positions and goals for all agents")
    .def("step",         &Environment::step,
        "Add actions, get observations, rewards and done")
    .def("render",       &Environment::render,
        "Show rendered image of the environment", py::arg("wait") = 0)
    .def("is_done",      &Environment::is_done,
        "Is the episode over")
    .def("get_number_of_agents", &Environment::get_number_of_agents,
        "Get number of agents")
    .def("get_episode_sim_time", &Environment::get_episode_sim_time,
        "How much time has passed in the simulation since the start of the episode")
    .def("get_observation_space", &Environment::get_observation_space,
        "Return shape of observation space (nagents x obs_len)")
    .def("get_action_space", &Environment::get_action_space,
        "Return shape of action space (nagents x act_len)");
}
