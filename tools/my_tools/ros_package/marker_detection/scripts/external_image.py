#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------




# rospy for the subscriber/publisher
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
# ==================== image subscriber ==========================
# Instantiate CvBridge
bridge = CvBridge()

def talker():
    pub = rospy.Publisher('marker_detection_image', Image, queue_size=10)
    rospy.init_node('external_image', anonymous=True)
    rate = rospy.Rate(0.1) # 10hz
    img_path = "../data/demo/test.png"
    while not rospy.is_shutdown():
        print "Reading image..."
        cv_image = cv2.imread(img_path)
        image_message = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
        pub.publish(image_message)
        rate.sleep()



if __name__ == '__main__':
   try:
       talker()
   except rospy.ROSInterruptException:
     pass