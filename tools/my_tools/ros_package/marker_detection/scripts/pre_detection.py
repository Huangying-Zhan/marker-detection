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
from std_msgs.msg import Bool
import cv2
import numpy as np
# ==================== image subscriber ==========================
# Instantiate CvBridge
bridge = CvBridge()

def talker():
    # define the publisher
    pub_signal = rospy.Publisher('detection_signal', Bool, queue_size=10)
    pub_image = rospy.Publisher('detection_image', Image, queue_size=10)
    # initialize ROS node
    rospy.init_node('pre_detection', anonymous=True)
    # read image
    rate = rospy.Rate(0.5) # 10hz
    img_path = "../data/demo/test.png"
    while not rospy.is_shutdown():
        # Set detection_signal for debugging purpose
        detection_signal = True
        print "Reading image..."
        # Convert image from OpenCV format to sensor_msgs.Image format
        cv_image = cv2.imread(img_path)
        detection_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        # Publish messages
        pub_signal.publish(detection_signal)
        pub_image.publish(detection_image)
        rate.sleep()



if __name__ == '__main__':
   try:
       talker()
   except rospy.ROSInterruptException:
     pass