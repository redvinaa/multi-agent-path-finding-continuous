#! /bin/bash
set -e
set -x

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "This will delete all runs that may exist, continue? [y/n]"
read ans

if [ "$ans" == "n" ]; then
    exit
fi

# delete all runs
sudo rm -rf ${SCRIPT_DIR}/multi_agent_sac/runs/*

# run training
param_groups=(empty_4x4 T_4x4)
for name in ${param_groups[@]}; do
    docker run \
        -v ${SCRIPT_DIR}/multi_agent_sac/runs:/opt/ros_ws/src/multi-agent-path-finding-continuous/multi_agent_sac/runs:Z \
        -v ${SCRIPT_DIR}/multi_agent_sac/params:/opt/ros_ws/src/multi-agent-path-finding-continuous/multi_agent_sac/params:Z \
        mapf bash rosrun multi_agent_sac training.py $name
done

# allow writing runs dir
sudo chmod 777 multi_agent_sac/runs/**

# generate plots for the runs
for name in ${param_groups[@]}; do
    rosrun multi_agent_sac generate_plots.py $name
done
