#!/usr/bin/env python

# ======================= Steps ========================
# list all annotation files
# list all images 
# get intersection as valid dataset
# use number of marker and number of repeat to copy prefix to trainval recursively
# 	for i in xrange(marker_num):
#	 	for j in xrange(repeat num):
# create trainval, train, val
# copy valid items to corresponding directories (duplicated annotation and images)
# rename object name in annotation files
# ======================================================

import os
from shutil import copyfile
import xml.etree.ElementTree as ET
import xml.dom.minidom as pt_xml
import shutil
import argparse

def parse_args():
	"""Parse input arguments."""
	parser = argparse.ArgumentParser(description='Marker dataset construction')
	parser.add_argument('--repeat_num', dest='repeat_num', help='number of repeatness of raw datast',
						default=5, type=int)
	args = parser.parse_args()
	return args

if __name__ == '__main__':
	raw_path = "./data/marker/marker_raw_data/"
	dst_path = "./data/marker/"
	raw_label = ['n04118538', 'n02802426', 'n02882301', 'n04254680', 'n03982232', 'n03134739', 'n03445777', 'n02839351']

	# number of repeated dataset
	args = parse_args()	
	dataset_dup_num = args.repeat_num 

	# find number of marker types
	marker_num = len(os.listdir(dst_path+"marker_img"))

	# Clear destination directories 
	shutil.rmtree(dst_path + "JPEGImages/")
	shutil.rmtree(dst_path + "ImageSets/")
	shutil.rmtree(dst_path + "Annotations/")
	os.makedirs(dst_path + "JPEGImages")
	os.makedirs(dst_path + "ImageSets")
	os.makedirs(dst_path + "Annotations")


	# Get trainval
	trainval = os.listdir(raw_path + "Annotations/")
	trainval = [item[:-4] for item in trainval]


	# Create duplicated trainval list
	new_trainval = []
	for i in xrange(marker_num):
		for j in xrange(dataset_dup_num):
			# example element in new_trainval: n01234567_123_marker_i_j
			extension = "_marker_" + str(i) + "_" + str(j)
			tmp_trainval = [k + extension for k in trainval]
			new_trainval += tmp_trainval


	# create trainval.txt
	trainval_dir = ''.join([dst_path, "ImageSets/", "trainval", ".txt"])
	f = open(trainval_dir,'w')
	for item in new_trainval:
		f.writelines(item+'\n')
	f.close()

	# create train.txt
	train_dir = ''.join([dst_path, "ImageSets/", "train", ".txt"])
	f = open(train_dir,'w')
	cnt = int(len(new_trainval)*0.8)
	train_set = new_trainval[:cnt]
	for item in train_set:
		f.writelines(item+'\n')
	f.close()

	# create val.txt
	val_dir = ''.join([dst_path, "ImageSets/", "val", ".txt"])
	f = open(val_dir,'w')
	val_set = new_trainval[cnt:]
	for item in val_set:
		f.writelines(item+'\n')
	f.close()

	# Copy valid imgs
	# dst_dir: destination directory
	dst_dir = ''.join([dst_path , "JPEGImages/"])
	cnt = 1
	for m in xrange(marker_num):
		for n in xrange(dataset_dup_num):
			for i in trainval:
				src = ''.join([raw_path, "JPEGImages/",  i, ".JPEG"])
				extension = "_marker_" + str(m) + "_" + str(n)
				dst = dst_dir + i + extension + ".JPEG"
				copyfile(src, dst)
				print cnt, '/', len(trainval)*marker_num*dataset_dup_num
				cnt += 1


	# Copy and modify valid xml
	# dst_dir: destination directory
	cnt = 1
	dst_dir = ''.join([dst_path , "Annotations/"])
	for prefix in trainval:
		for m in xrange(marker_num):
			for n in xrange(dataset_dup_num): 
				# Open original file
				xml_src = ''.join([raw_path, "Annotations/",  prefix, ".xml"])
				tree = ET.parse(xml_src)
				root = tree.getroot()
				# Update xml
				# update filename
				extension = "_marker_" + str(m) + '_' + str(n)
				root.find('filename').text = ''.join([prefix, extension])
				# update name in objects
				objs = root.findall('object')
				for obj in objs:
					bbox_name = obj.find('name').text 
					if bbox_name in raw_label:
						obj.find('name').text = ''.join(["marker_", str(m)])
				# Write back to file
				extension = "_marker_" + str(m) + '_' + str(n)
				dst = dst_dir + prefix + extension + '.xml'
				tree.write(dst)
				print cnt, '/', len(trainval)*marker_num*dataset_dup_num
				cnt += 1