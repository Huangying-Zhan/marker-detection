#!/usr/bin/env python
import os
import os.path as osp
import sys

# def init_path():
#     def add_path(path):
#         if path not in sys.path:
#             sys.path.insert(0, path)
#     this_dir = os.getcwd()
#     # Add caffe to PYTHONPATH
#     caffe_path = osp.join(this_dir, '..', 'caffe-fast-rcnn', 'python')
#     add_path(caffe_path)
#     # Add lib to PYTHONPATH
#     lib_path = osp.join(this_dir, '..', 'lib')
#     add_path(lib_path)
#     # Add ros lib to PYTHONPATH
#     ros_lib_path = osp.join(this_dir, 'devel/lib/python2.7/dist-packages')
#     add_path(ros_lib_path)

# init_path()

"""import neccesary libraries """
import numpy as np
import rospy
from marker_detection.msg import marker_detection_result as md_result
from marker_detection.msg import bbox as bbox
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import shutil, os
import os.path as osp
import cv2

# Re-make a folder for saved images
# saved_img_path = "./image_result_debug/"
# if osp.exists(saved_img_path):
#     shutil.rmtree(saved_img_path)
#     os.makedirs(saved_img_path)
# else:
#     os.makedirs(saved_img_path)

# ====================== result subscriber =================
def num_result_callback(msg):
    marker_detected = msg.marker_detected
    if marker_detected:
        print("Received detection numerical result!")
        prob = msg.prob
        bboxes = msg.bboxes
        print bboxes[0].bbox
    else:
        print "Marker is not detected..."

# Instantiate CvBridge
bridge = CvBridge()

def image_result_callback(msg):
    print("Received detection image result!")
    # Convert your ROS Image message to OpenCV2
    cv2_img = bridge.imgmsg_to_cv2(msg, "8UC3")
    
    # save subscribed image
    # num_saved_img = len(os.listdir(saved_img_path))
    # cv2.imwrite(saved_img_path + str(num_saved_img) + ".png", cv2_img)

def combined_result_callback(msg):
    marker_detected = msg.marker_detected
    if marker_detected:
        print("Received detection result!")
        prob = msg.prob
        bboxes = msg.bboxes
        detection_image_result = msg.detection_image_result
        print bboxes[0].bbox
        image_talker(detection_image_result)
    else:
        print "Marker is not detected..."

# publish image result
def image_talker(cv_image):
    # Obtaining detection result
    pub.publish(cv_image)
    print "Published image result!"
    

if __name__ == '__main__':
    # Setup ROS
    rospy.init_node('result_listener')
    # Set up numerical result subscriber and define its callback
    # rospy.Subscriber("detection_num_result", md_result, num_result_callback)
    # Set up image result subscriber and define its callback
    # rospy.Subscriber("detection_image_result", Image, image_result_callback)
    # Set up combined result subscriber and define its callback
    rospy.Subscriber("detection_result", md_result, combined_result_callback)
    # setup image publisher
    pub = rospy.Publisher('detection_image_result', Image, queue_size=1)
    # Spin until ctrl + c
    rospy.spin()
