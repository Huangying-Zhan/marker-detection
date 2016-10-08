mkdir -p catkin_ws/src
cd catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
source devel/setup.bash

# Create a ROS package
# pwd: catkin_ws
cp -r ../tools/my_tools/ros_package/marker_detection/ ./src/marker_detection/

# Prepare publisher
catkin_make
catkin_make install