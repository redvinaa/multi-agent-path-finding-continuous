FROM ros:melodic

# install build tools
RUN apt-get update
RUN apt-get install -y  python-catkin-tools libeigen3-dev
RUN apt-get install -y  ros-melodic-image-transport ros-melodic-cv-bridge ros-melodic-pybind11-catkin
RUN rm -rf /var/lib/apt/lists/*

# install box2d
RUN git -C / clone https://github.com/erincatto/box2d.git
RUN mkdir -p /box2d/build
WORKDIR /box2d/build
RUN cmake ..
RUN make install


# clone ros package repo
ENV ROS_WS /opt/ros_ws
WORKDIR $ROS_WS
RUN mkdir -p $ROS_WS/src
COPY . src

# build ros package source
RUN catkin config \
      --extend /opt/ros/$ROS_DISTRO && \
    catkin build \
      mapf_environment

# source ros package from entrypoint
RUN sed --in-place --expression \
      '$isource "$ROS_WS/devel/setup.bash"' \
      /ros_entrypoint.sh

# run ros package launch file
CMD ["rosrun", "mapf_environment", "environment"]
