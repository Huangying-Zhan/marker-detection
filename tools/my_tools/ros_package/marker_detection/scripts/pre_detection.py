#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import os
import os.path as osp
import sys

def init_path():
    def add_path(path):
        if path not in sys.path:
            sys.path.insert(0, path)
    this_dir = os.getcwd()
    # Add ros lib to PYTHONPATH
    ros_lib_path = osp.join(this_dir, 'devel/lib/python2.7/dist-packages')
    add_path(ros_lib_path)

init_path()


# rospy for the subscriber/publisher
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
from marker_detection.msg import pre_detection as pre_det_msg
import cv2
import numpy as np
# ==================== image subscriber ==========================
# Instantiate CvBridge
bridge = CvBridge()

def camera_image_callback(msg):
    # print("Received detection image result!")
    # Convert your ROS Image message to OpenCV2
    print "Publishing pre-detection message..."
    pub = rospy.Publisher('pre_detection', pre_det_msg, queue_size=1)
    msg_to_send = pre_det_msg()
    msg_to_send.detection_image = msg
    msg_to_send.detection_signal = True
    pub.publish(msg_to_send)

def talker():
    # define the publisher
    pub = rospy.Publisher('pre_detection', pre_det_msg, queue_size=1)
    # initialize ROS node
    rospy.init_node('pre_detection', anonymous=True)
    # read image
    rate = rospy.Rate(0.5) # 10hz
    img_path = "../data/demo/test.png"
    while not rospy.is_shutdown():
        msg_to_send = pre_det_msg()
        # Set detection_signal for debugging purpose
        msg_to_send.detection_signal = True
        print "Reading image..."
        # Convert image from OpenCV format to sensor_msgs.Image format
        cv_image = cv2.imread(img_path)
        msg_to_send.detection_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        # Publish messages
        pub.publish(msg_to_send)
        rate.sleep()



if __name__ == '__main__':
    rospy.init_node('pre_marker_detection')
    # Set up your subscriber and define its callback
    rospy.Subscriber("/camera/image_color", Image, camera_image_callback)
    # Spin until ctrl + c
    rospy.spin()
