# Multiagent path-finding (continuous)

### Usage:
This repository containes 3 packages, intended for use with ROS Melodic.  
All of this is dockerized, so if you have docker installed, all you need to  
do is run`bash build_container.bash` to build the docker image and  
`bash train_in_container.bash` to run the training. Parameters for the training  
can be found in multi_agent_sac/params. These are shared as a volume to docker  
so you don't need to rebuild the images after changing the parameters. The  
measured variables can be tracked throughout training using tensorboard  
(`pip3 install tensorboard`). The logdir is multi_agent_sac/runs, which  
is a volume as well. Doxygen documentation can be build using the Doxyfile  
(just run `doxygen` in the root of the repo).

### Packages:
 - **mapf_environment**: This contains the C++ code for the simulator.
 - **mapf_environment_py**: This is the Python3 wrapper for the simulator.
 - **multi_agent_sac**: This is the multi-agent SAC implementation.
