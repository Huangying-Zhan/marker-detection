mkdir -p catkin_ws/src
cd catkin_ws/src
catkin_init_workspace
cd ..
catkin_make
source devel/setup.bash

# Create a ROS package
cd src/
catkin_create_pkg marker_detection std_msgs rospy cv_bridge sensor_msgs message_generation
cd ..
catkin_make
. ./devel/setup.sh

# Update package.xml and CMakeLists.txt
cd src/marker_detection
cp ../../../tools/my_tools/ros_package/marker_detection/CMakeLists.txt ./
cp ../../..//tools/my_tools/ros_package/marker_detection/package.xml ./

# Create new message
roscd marker_detection
mkdir msg
echo -e "int32[4] bbox" > msg/bbox.msg
echo -e "bool marker_detected\nfloat32[] prob\nbbox[] bboxes" > msg/marker_detection_result.msg

# Prepare publisher
# roscd marker_detection
cd ../../../
cp -r ./tools/my_tools/ros_package/marker_detection/scripts/ ./catkin_ws/src/marker_detection/scripts/

# Update number of CLASSES at scripts/marker_detection_ros.py
# make
cd ./catkin_ws
catkin_make
catkin_make install