#!/usr/bin/env python

# ====================== Idea =========================
# For each image (Img):
# 	Get valid bounding box regions (R) in Img. (Annotation files provide the information)
# 	For each region (rgn) in R:
# 		1. Transform marker image (M) randomly (rotation, shear, scaling, flip). 
# 			Call transformed marker image as M'
# 		2. Get size of rgn (height & width)
# 		3. Obtain new marker image (M'') by resizing M' into the size of rgn.
# 		4. Replace rgn by M''.
# 	Randomly change the brightness of the modified Img.
# 	Save the modified Img.

# =====================================================


import xml.etree.ElementTree as ET
from ImageAugmenter import ImageAugmenter
from scipy import misc
import numpy as np
import cv2
from PIL import ImageEnhance
from PIL import Image
import cv2
import random as rand
import os

data_path = "./data/marker/"
annotation_path = data_path+"Annotations/"
marker_dir = data_path + "marker_img/"

marker_list = os.listdir(marker_dir)
marker_list.sort()
marker_paths = [marker_dir + i for i in marker_list]


def BrightnessEnhancer(raw_img, RBG_BGR = 1 , gauss_mu = 1, gauss_sd = 0.5 , brightness_fac_min = 0.3, brightness_fac_max = 2.5):
    img_enhancer = ImageEnhance.Brightness(raw_img)
    rand_fac = min(brightness_fac_max, max(brightness_fac_min, int(rand.gauss(gauss_mu, gauss_sd)*10)/10.0))
    img = img_enhancer.enhance(rand_fac)
    # Depends on raw_img reading source (option 1: PIL.Image ; option 2: cv2)
    # Option 1: PIL.Image.open(), BGR array
    if RBG_BGR == 1:
    	enhanced_img = np.array(img)
    elif RBG_BGR == 2:
    	enhanced_img = np.array(img)[:,:,[2,1,0]]
    return enhanced_img

def PartialBrightnessEnhancer(raw_img, RBG_BGR = 1):
	# enhance brightness of partial image
    enhanced_img = BrightnessEnhancer(raw_img, RBG_BGR)
    enhanced_img_region = BrightnessEnhancer(raw_img, RBG_BGR)
    # random brightness region
    img = enhanced_img  
    img_size = [img.shape[0], img.shape[1]]
    rand_fac_1 = rand.random()
    rand_fac_2 = rand.random()
    if rand_fac_1>rand_fac_2:
    	rand_fac_1, rand_fac_2 = rand_fac_2, rand_fac_1
    y_min = int(img_size[0] * rand_fac_1)
    y_max = int(img_size[0] * rand_fac_2)
    x_min = int(img_size[1] * rand_fac_1)
    x_max = int(img_size[1] * rand_fac_2)
    img[y_min:y_max,x_min:x_max] = enhanced_img_region[y_min:y_max,x_min:x_max]
    return (img)

if __name__ == '__main__':

	# Read trainval.txt
	trainval = data_path + "ImageSets/trainval.txt"
	f = open(trainval,'r')
	s=f.readlines()
	trainval=[]
	for i in s:
		trainval.append(i[:-1])

	# Read marker picture
	markers = []
	for path in marker_paths:
		markers.append(cv2.imread(path))
	cnt = 1
	for item in trainval:
		# print counter
		if cnt % 100 == 0:
			print cnt,"/",len(trainval)
		cnt += 1
		# select correct marker pic
		for i in xrange(len(marker_paths)):
			if "marker_"+str(i) in item:
				marker = markers [i]
			else:
				continue
		# read dataset image
		pic_path = data_path + "JPEGImages/" + item + ".JPEG"
		pic = cv2.imread(pic_path)
		pic_height = pic.shape[0]
		pic_width = pic.shape[1]
		xml_path = annotation_path + item + ".xml"
		annotation = ET.parse(xml_path).getroot()
		objs = annotation.findall('object')
		for ix, obj in enumerate(objs):
			# read bboxq
		    bbox = obj.find('bndbox')
		    x1 = int(bbox.find('xmin').text)
		    y1 = int(bbox.find('ymin').text)
		    x2 = int(bbox.find('xmax').text)
		    y2 = int(bbox.find('ymax').text)
		    # limit bbox to not outside original pic
		    if x1>pic_width or x2>pic_width or y1>pic_height or y2>pic_height:
		    	continue
		    bbox = [x1,y1,x2,y2]
		    # resize marker
		    bbox_shape = [x2-x1, y2-y1]
		    tmp_marker = cv2.resize(marker, (bbox_shape[0],bbox_shape[1]))
		    # Image augmentation
		    height = tmp_marker.shape[0]
		    width = tmp_marker.shape[1]
		    augmenter = ImageAugmenter(width, height, # width and height of the image (must be the same for all images in the batch)
										hflip=True,    # flip horizontally with 50% probability
										vflip=True,    # flip vertically with 50% probability
										scale_to_percent=1.5, # scale the image to 70%-130% of its original size
										scale_axis_equally=False, # allow the axis to be scaled unequally (e.g. x more than y)
										rotation_deg=45,    # rotate between -25 and +25 degrees
										shear_deg=20,       # shear between -10 and +10 degrees
										translation_x_px=0, # translate between -5 and +5 px on the x-axis
										translation_y_px=0  # translate between -5 and +5 px on the y-axis
										)
			# augment a batch containing only this image
			# the input array must have dtype uint8 (ie. values 0-255), as is the case for scipy's imread()
			# the output array will have dtype float32 (0.0-1.0) and can be fed directly into a neural network
		    tmp_marker = augmenter.augment_batch(np.array([tmp_marker], dtype=np.uint8))
		    # Convert tmp_marker back to uint8 format
		    tmp_marker = tmp_marker[0] * 255
		    tmp_marker=tmp_marker.astype(np.uint8)
		    # replace original bounding region by marker pic
		    pic[y1:y2,x1:x2] = tmp_marker
		    # enhance whole image
		    pic = Image.fromarray(pic)
		    pic = BrightnessEnhancer(pic)
		cv2.imwrite(pic_path,pic)
