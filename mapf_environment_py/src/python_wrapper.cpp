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
    .def(py::init<std::string, std::tuple<float, float>, int, unsigned int, int, float, float, float, int>(),
        py::arg("map_path"),
        py::arg("map_size"),
        py::arg("number_of_agents")          = 2,
        py::arg("seed")                      = 0,
        py::arg("max_steps")                 = 30,
        py::arg("robot_diam")                = 0.8,
        py::arg("noise")                     = 0.01,
        py::arg("physics_step_size")         = 0.1,
        py::arg("step_multiply")             = 10
        )
    .def("reset",        &Environment::reset,
        "Set done=false, generate new starting positions and goals for all agents")
    .def("step",         &Environment::step,
        "Add actions, get observations, rewards and done", py::arg("action"), py::arg("render") = false)
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
