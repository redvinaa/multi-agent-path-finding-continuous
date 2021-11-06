FROM ros:melodic

# install build tools
RUN apt-get update
RUN apt-get install -y cmake libx11-dev xorg-dev libglu1-mesa-dev wget zip unzip
RUN apt-get install -y python-catkin-pkg python-catkin-tools
RUN apt-get install -y ros-melodic-image-transport ros-melodic-cv-bridge ros-melodic-pybind11-catkin
RUN apt-get install -y python3-pip
RUN pip3 install catkin_pkg torch matplotlib tensorboard tensorboardX rospkg tqdm

# install box2d
RUN git -C / clone https://github.com/erincatto/box2d.git
RUN mkdir -p /box2d/build
WORKDIR /box2d/build
RUN cmake ..
RUN make install
WORKDIR /

# install eigen
RUN wget https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip -P /
RUN unzip eigen-3.3.7.zip
RUN mkdir -p /eigen-3.3.7/build
WORKDIR /eigen-3.3.7/build
RUN cmake ..
RUN make install
WORKDIR /


# clone this repo into new ros workspace
ENV ROS_WS /opt/ros_ws
WORKDIR $ROS_WS
RUN mkdir -p $ROS_WS/src
COPY . src

# build ros packages
RUN catkin config --extend /opt/ros/$ROS_DISTRO && catkin build
ENV PYTHONPATH $PYTHONPATH:/opt/ros_ws/devel/lib
ENV PYTHONPATH $PYTHONPATH:/opt/ros_ws/devel/lib/python3/dist-packages
ENV PYTHONPATH $PYTHONPATH:/opt/ros_ws/src/multi-agent-path-finding-continuous

# source ros package from entrypoint
RUN sed --in-place --expression \
      '$isource "$ROS_WS/devel/setup.bash"' \
      /ros_entrypoint.sh

# expose port 6006 for tensorboard
EXPOSE 6006

# run ros package launch file
CMD ["/bin/bash"]
