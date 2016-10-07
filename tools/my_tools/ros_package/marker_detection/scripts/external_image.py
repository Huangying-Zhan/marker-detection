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
from marker_detection.msg import pre_detection as pre_det
import cv2
import numpy as np
# ==================== image subscriber ==========================
# Instantiate CvBridge
bridge = CvBridge()

def talker():
    pub = rospy.Publisher('pre_marker_detection', pre_det, queue_size=10)
    rospy.init_node('external_image', anonymous=True)
    rate = rospy.Rate(0.5) # 10hz
    img_path = "../data/demo/test.png"
    while not rospy.is_shutdown():
        pre_det_msg = pre_det()
        pre_det_msg.detection_signal = True
        print "Reading image..."
        cv_image = cv2.imread(img_path)
        # pre_det_msg.camera_image = Image()
        pre_det_msg.camera_image = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        pub.publish(pre_det_msg)
        rate.sleep()



if __name__ == '__main__':
   try:
       talker()
   except rospy.ROSInterruptException:
     pass