#! /bin/bash

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

docker run \
    -v ${SCRIPT_DIR}/multi_agent_sac/runs:/opt/ros_ws/src/multi-agent-path-finding-continuous/multi_agent_sac/runs \
    -v ${SCRIPT_DIR}/multi_agent_sac/params:/opt/ros_ws/src/multi-agent-path-finding-continuous/multi_agent_sac/params \
    mapf bash rosrun multi_agent_sac training.py $*
