#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Faster R-CNN."""

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
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import caffe, os, sys, cv2
import re
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import shutil
from marker_detection.msg import marker_detection_result as md_result

ros_root = os.getcwd()
FRCN_root = ros_root + "/../"

CLASSES = ('__background__',
           'marker_0')


# Re-make a folder for saved images
saved_img_path = ros_root + "/src/marker_detection/detected_img/"
if osp.exists(saved_img_path):
    shutil.rmtree(saved_img_path)
    os.makedirs(saved_img_path)
else:
    os.makedirs(saved_img_path)


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    
    num_saved_img = len(os.listdir(saved_img_path))

    # Case that nothing detected
    if len(inds) == 0:
        cv2.imwrite(saved_img_path + str(num_saved_img) + ".png", im)
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.savefig(saved_img_path + str(num_saved_img) + ".png")
 

def detection(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]                
        vis_detections(im, cls, dets, thresh=CONF_THRESH)
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    return dets[inds]

# ====================== result publisher =================
def talker(dets):
    pub = rospy.Publisher('marker_detection_result', md_result, queue_size=10)
    # rospy.init_node('talker', anonymous=True)
    # Obtaining detection result
    msg_to_send = md_result()
    if len(dets) != 0:
        msg_to_send.marker_detected = True
        # Convert dets to valid message
        msg_to_send.prob = list(dets[:,-1])
        tmp_bboxes = list(dets[:,:4])
        bbox = [i for bbox in tmp_bboxes for i in bbox]
        msg_to_send.bbox = [int(i) for i in bbox]
    pub.publish(msg_to_send)
    print "Publishing result completed!"

# ==================== image subscriber ==========================
# Instantiate CvBridge
bridge = CvBridge()

def image_callback(msg):
    print("Received an image!")
    # cpu mode
    caffe.set_mode_cpu()
    # gpu mode
    # caffe.set_mode_gpu()
    # caffe.set_device(args.gpu_id)
    # cfg.GPU_ID = args.gpu_id
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(msg, "8UC3")
        dets = detection(net,cv2_img)
        print "detection completed!"
        # Publish result
        talker(dets)
    except CvBridgeError, e:
        print(e)




if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = FRCN_root + "models/marker/test.prototxt"
    caffemodel = FRCN_root + "output/marker/train/zf_faster_rcnn_marker_iter_50000.caffemodel"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Setup ROS
    rospy.init_node('image_listener')
    # Define your image topic
    topic = "marker_detection_image"
    # Set up your subscriber and define its callback
    rospy.Subscriber(topic, Image, image_callback)
    # Spin until ctrl + c

    rospy.spin()


