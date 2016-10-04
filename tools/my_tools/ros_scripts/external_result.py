#!/usr/bin/env python
import os
import os.path as osp
import sys

def init_path():
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    this_dir = os.getcwd()
    # Add caffe to PYTHONPATH
    caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')
    add_path(caffe_path)
    # Add lib to PYTHONPATH
    lib_path = osp.join(this_dir, '..', 'lib')
    add_path(lib_path)
    # Add ros lib to PYTHONPATH
    ros_lib_path = osp.join(this_dir, 'devel/lib/python2.7/dist-packages')
    add_path(ros_lib_path)

init_path()

"""import neccesary libraries """
import numpy as np
import rospy
from marker_detection.msg import marker_detection_result as md_result

# ====================== result publisher =================
def result_callback(msg):
    print("Received detection result!")
    # Convert your ROS Image message to OpenCV2
    marker_detected = msg.marker_detected
    if marker_detected:
        prob = msg.prob
        bbox = msg.bbox
        print bbox

if __name__ == '__main__':
    # Setup ROS
    rospy.init_node('result_listener')
    # Define your image topic
    topic = "marker_detection_result"
    # Set up your subscriber and define its callback
    rospy.Subscriber(topic, md_result, result_callback)
    # Spin until ctrl + c

    rospy.spin()
