#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------


import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import time
import os

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='zf')
    parser.add_argument('--mode', dest='mode', 
                        help='There are 2 modes.\n 0: Export mode (only export numerical infomation, including bbox and depth)  \n 1: Visualizationn',
						choices=[0,1,2],default=0, type=int)
    parser.add_argument('--class', dest='CLS_IND', 
                        help='There are 20 possibleclasses, from 1 to 20',
                        default = 15, type=int)
    parser.add_argument('--image_source', dest='IMG_SRC', 
                        help='0: Online Mode (load same image) or 1: Offline mode (load a set of images)',
                        default = 1, type=int)
    args = parser.parse_args()
    return args


def vis_detections(dist, plt_cnt, FIG, im, class_name, bboxes, scores, thresh = 0.5):
    """Draw detected bounding boxes."""
    row = 3
    col = 3
    im = im[:, :, (2, 1, 0)] #RGB -> BGR
    FIG.add_subplot(row, col, plt_cnt)
    plt.cla()
    ax = plt.gca()
    ax.imshow(im, aspect='equal')
    for i in xrange(bboxes.shape[0]):
        bbox = bboxes[i]
        score = scores[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]), # bottom left corner
                          bbox[2] - bbox[0], # width
                          bbox[3] - bbox[1], # height
                          fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score) + "\ndist: " +str(dist[i]) ,
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')


    # ax.set_title(('{} detections with '
    #               'p({} | box) >= {:.1f}').format(class_name, class_name,
    #                                               thresh),
    #               fontsize=14)
    plt.axis('off')
    plt.draw()
    plt.pause(0.0005)
    print "PLT_CNT: ", plt_cnt



def read_depth_file(depth_map_name):
  # # Important! return a depth_map as np.array
  # read depth map and reshape to (height, width)
  
  depth_map = cv2.imread(depth_map_name,-1)
  # depth_map.dtype = np.float32
  depth_map.astype(float)
  depth_map = depth_map.reshape((depth_map.shape[0], depth_map.shape[1]))

  # Convert nan and inf elements to zero
  depth_map[np.isnan(depth_map)] = 0
  depth_map[np.isinf(depth_map)] = 0
  return depth_map


def get_dist_from_depth_map(bboxes, depth_map, img_shape):
  # Input is bboxes's coordinate and depth_map as a np.array got from read_depth_file()
  center_ratio = 0.3
  dist = []
  resize_fac = depth_map.shape[0] / img_shape[0]
  for bbox in bboxes:
    # Get coordinates for 2 corners of restored image
    x1 = int(bbox[0] * resize_fac) # top_left_x
    y1 = int(bbox[1] * resize_fac) # top_left_y
    x2 = int(bbox[2] * resize_fac) # bottom_right_x
    y2 = int(bbox[3] * resize_fac) # bottom_right_y
    
    # Calculate the restored height and width of bbox
    bbox_height = y2 - y1
    bbox_width = x2 - x1

    # Calculate the height and width of center window
    center_height = max(1, int(bbox_height * center_ratio))
    center_width = max(1, int((bbox_width) * center_ratio))

    # Find out the start_row
    start_row = y1 + ((bbox_height - center_height) / 2 )
    end_row = start_row + center_height
    start_col = x1 + ((bbox_width - center_width) / 2)
    end_col = start_col + center_width

    # Calculate mean distance of center region
    center_depth_map = depth_map[start_row:end_row, start_col:end_col]
    center_depth_map = center_depth_map[center_depth_map!=0] # invalid distance (nan, inf, -inf are all regards as zero)
    if len(center_depth_map)!=0:
      dist.append(int(np.average(center_depth_map)))
    else: 
      dist.append("N/A")
  return dist # dist with length = number of objects


def detection(net, im):
    # Forward pass and count time
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.diff, boxes.shape[0])
    
    # Return valid bbox and score of the requested class
    cls_boxes = boxes[:, 4*CLS_IND:4*(CLS_IND + 1)]
    cls_scores = scores[:, CLS_IND]
    dets = np.hstack((cls_boxes,
                      cls_scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, NMS_THRESH)
    dets = dets[keep, :]
    inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
    if len(inds) == 0:
      return np.asarray([0]),np.asarray([0])
    final_bbox = dets[inds, :4]
    final_scores = dets[inds, -1]
    return final_bbox, final_scores


# =====================Global variable setup=================================
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

CONF_THRESH = 0.8 # A threshold used to remove bbox with score less than it.
NMS_THRESH = 0.2 # non-maxima suppression threshold, which is used to remove duplicate bbox

# directories for captured images and depth map
rgb_img_dir = "/home/ubuntu/Desktop/zed_proj/zed_save_imgs/captured_image/Image/"
depth_map_dir = "/home/ubuntu/Desktop/zed_proj/zed_save_imgs/captured_image/DepthMap/"
rgb_img_list = os.listdir(rgb_img_dir)
rgb_img_list.sort()
depth_map_list = os.listdir(depth_map_dir)
depth_map_list.sort()

# Reading parser arguments
args = parse_args()

prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                        'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                          NETS[args.demo_net][1])

MODE = args.mode
CLS_IND = args.CLS_IND
IMG_SRC = args.IMG_SRC

if not os.path.isfile(caffemodel):
    raise IOError(('{:s} not found.\nDid you run ./data/script/'
                   'fetch_faster_rcnn_models.sh?').format(caffemodel))

if args.cpu_mode:
    caffe.set_mode_cpu()
else:
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    cfg.GPU_ID = args.gpu_id

# ------------------------ main function ------------------------


if __name__=='__main__':
  valid_cnt = 0
  invalid_cnt = 0
  # Use RPN for proposals
  cfg.TEST.HAS_RPN = True 


  # initialize net
  net = caffe.Net(prototxt, caffemodel, caffe.TEST)
  print '\n\nLoaded network {:s}'.format(caffemodel)

  class_name = CLASSES[CLS_IND]
  exist_img_time = 0
  offline_mode_cnt = 0

  # Pre-setup for Visualization mode,Plot a blank figure for later use
  if MODE == 1:
    print "FIG.show()"
    FIG = plt.figure(figsize=(48,32))
    FIG.show()
    plt.pause(0.005)
    plt_cnt = 1

  for i in xrange(len(rgb_img_list)):
    rgb_img_name = rgb_img_dir + rgb_img_list[i]
    depth_map_name = depth_map_dir + depth_map_list[i]

  timer = Timer()
  while offline_mode_cnt < len(depth_map_list):
    try:
      # Before Detection
      # Image source mode: Online mode (load same image)
      if IMG_SRC == 0:
        rgb_img_name = rgb_img_dir + "ZEDImage.png"
        depth_map_name = depth_map_dir + "ZEDDepthMap.png"
        new_img_time = time.ctime(os.path.getmtime(rgb_img_name))
        if new_img_time == exist_img_time:
          continue
        else:
          exist_img_time = new_img_time

      # Image source mode: Offline mode (load a set of images)
      if IMG_SRC == 1:
        rgb_img_name = rgb_img_dir + rgb_img_list[offline_mode_cnt]
        depth_map_name = depth_map_dir + depth_map_list[offline_mode_cnt]
        offline_mode_cnt += 1

      # read image
      img = cv2.imread(rgb_img_name)
      img_shape = [img.shape[0], img.shape[1]]

      # do detection
      final_bbox, final_scores = detection (net, img)

      # get distance of objects from depth_map
      depth_map = read_depth_file(depth_map_name)
      dist = get_dist_from_depth_map (final_bbox, depth_map, img_shape)


      # After detection
      # Check if there exist objects
      if len(final_bbox.shape)!=1:
        # Mode 0: Export bbox and corresponding scores
        valid_cnt+=1
        print "Valid detection count: "valid_cnt
        if MODE == 0:
          print "bbox: \n", final_bbox
          print "dist: ", dist, "\n"
          continue

        # Mode 1: Visualizing detection result
        if MODE == 1:
          timer.tic()
          vis_detections(dist, plt_cnt, FIG, img, class_name, final_bbox, final_scores, CONF_THRESH)
          timer.toc()
          print ('Detection took {:.3f}s for visualization').format(timer.diff)
          plt_cnt += 1
          if plt_cnt == 10:
            plt_cnt = 1
          continue

    except IndexError: pass
    except IOError: pass
    except SyntaxError: pass
    except AttributeError: pass
  raw_input("Press Enter to continue...")


