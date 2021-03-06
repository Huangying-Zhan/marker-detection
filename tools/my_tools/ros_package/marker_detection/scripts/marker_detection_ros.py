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
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
# General
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil
import PIL 
from PIL import ImageFont
from PIL import Image as PIL_Image
from PIL import ImageDraw

# ROS
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from cv_bridge import CvBridge, CvBridgeError
from marker_detection.msg import pre_detection as pre_det_msg
from marker_detection.msg import marker_detection_result as md_result
from marker_detection.msg import bbox as bbox_msg



ros_root = os.getcwd()
FRCN_root = ros_root + "/../"

CLASSES = ('__background__',
           'marker')


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

    # Case that nothing detected
    num_saved_img = len(os.listdir(saved_img_path))
    if len(inds) == 0:
        cv2.imwrite(saved_img_path + str(num_saved_img) + ".png", im)
        # publish image message
        image_talker(im)
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    # canvas = FigureCanvas(fig)
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
    plt.close()
    # publish image message
    img = cv2.imread(saved_img_path + str(num_saved_img) + ".png")
    image_talker(img)

def resize_PIL_image(img, basewidth):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    return img

def resize_CV_image(img, basewidth):
    wpercent = (basewidth / float(img.shape[0]))
    hsize = int((float(img.shape[1]) * float(wpercent)))
    img = cv2.resize(img, (hsize, basewidth)) 
    return img


def fast_vis_detections(cv_im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]

    # Case that nothing detected
    num_saved_img = len(os.listdir(saved_img_path))
    if len(inds) == 0:
        # cv_im = resize_CV_image(cv_im, 300)
        # cv2.imwrite(saved_img_path + str(num_saved_img) + ".png", cv_im)
        # publish image message
        # image_talker(cv_im)
        return cv_im

    # change from OpenCV format to PIL Image format
    array_im = cv_im[:, :, (2, 1, 0)]
    PIL_im = PIL_Image.fromarray(array_im)
    # Draw bbox, text
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 30)
    for i in inds:
        # Get bbox and score
        bbox = dets[i, :4]
        score = dets[i, -1]
        # Draw bbox and text
        draw = ImageDraw.Draw(PIL_im)
        draw.rectangle(bbox, outline = (0,0,255))
        draw.text((0,0),'{} detections with p({} | box) >= {:.1f}'.format(class_name, class_name,thresh), (255,0,0), font = font)
        draw.text((bbox[:2]), '{:s}: {:.3f}'.format(class_name, score),(0,0,255), font = font)
        draw = ImageDraw.Draw(PIL_im)
    # Save locally
    # PIL_im = resize_PIL_image(PIL_im,300)
    # PIL_im.save(saved_img_path + str(num_saved_img) + ".png")
    # Convert im back to OpenCV format for publishing purpose
    cv_im = np.asarray(PIL_im)[:, :, (2, 1, 0)]
    # publish image message
    # image_talker(cv_im)
    return cv_im
 

def detection(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for {:d} object proposals').format(timer.total_time, boxes.shape[0])

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
        cv_im_result = fast_vis_detections(im, cls, dets, thresh=CONF_THRESH)
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    return dets[inds], cv_im_result

# ====================== numerical result publisher =================
def num_talker(dets):
    pub = rospy.Publisher('detection_num_result', md_result, queue_size=1)
    # Obtaining detection result
    msg_to_send = md_result()
    if len(dets) != 0:
        msg_to_send.marker_detected = True
        # Convert dets to valid message
        msg_to_send.prob = dets[:,-1].tolist()
        bboxes = dets[:,:4].astype(np.int32).tolist()
        bboxes_to_send = []
        for box in bboxes:
            bboxes_to_send.append(bbox_msg(box))
        msg_to_send.bboxes = bboxes_to_send
    pub.publish(msg_to_send)
    print "Published numerical result!"

# ====================== image result publisher =================
def image_talker(cv_image):
    pub = rospy.Publisher('detection_image_result', Image, queue_size=1)
    # Obtaining detection result
    msg_to_send = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
    pub.publish(msg_to_send)
    print "Published image result!"

# ====================== combined result publisher =================
def result_talker(dets, cv_image):
    # Obtaining detection result
    msg_to_send = md_result()
    if len(dets) != 0:
        msg_to_send.marker_detected = True
        # Convert dets to valid message
        msg_to_send.prob = dets[:,-1].tolist()
        bboxes = dets[:,:4].astype(np.int32).tolist()
        bboxes_to_send = []
        for box in bboxes:
            bboxes_to_send.append(bbox_msg(box))
        msg_to_send.bboxes = bboxes_to_send
    # image result
    detection_image_result = bridge.cv2_to_imgmsg(cv_image, encoding="passthrough")
    msg_to_send.detection_image_result = detection_image_result
    pub.publish(msg_to_send)
    print "Published combined (numerical + image) result!"

# ==================== Subscriber ==========================
# Instantiate CvBridge
bridge = CvBridge()


def pre_detection_callback(msg):
    # cpu mode
    caffe.set_mode_cpu()
    # Get detection_signal
    detection_signal = msg.detection_signal
    try:
        # Convert your ROS Image message to OpenCV2
        # detection_signal = pre_detection_signal_callback()
        if detection_signal:
            cv2_img = bridge.imgmsg_to_cv2(msg.detection_image, "8UC3")
            # if detection_signal:
            print("Start detection!")
            timer = Timer()
            timer.tic()
            dets, cv_im_result = detection(net,cv2_img)
            # Publish result
            result_talker(dets, cv_im_result)
            print "Detection completed!"
            timer.toc()
            print "Total time: ", timer.total_time
            print "\n\n"
    except CvBridgeError, e:
        print(e)




if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    prototxt = FRCN_root + "models/marker/test.prototxt"
    caffemodel_at_ros = os.listdir(FRCN_root+ "output/marker/train/ros/")
    if len(caffemodel_at_ros) == 1:
        caffemodel = FRCN_root + "output/marker/train/ros/" + caffemodel_at_ros[0]
    else: 
        print "Reading caffemodel error. Is there only ONE caffemodel saved at $FRCN/output/marker/train/ros/ ?"

    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))

    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)

    # Setup ROS
    rospy.init_node('marker_detection')
    # Set up your subscriber and define its callback
    rospy.Subscriber("pre_detection", pre_det_msg, pre_detection_callback)
    # set up publisher
    pub = rospy.Publisher('detection_result', md_result, queue_size=1)
    # Spin until ctrl + c
    rospy.spin()


