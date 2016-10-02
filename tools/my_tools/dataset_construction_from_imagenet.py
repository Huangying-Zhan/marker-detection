import os
from shutil import copyfile
import xml.etree.ElementTree as ET
import xml.dom.minidom as pt_xml

# Steps
# list all annotation files
# list all images 
# get intersection as valid dataset
# use number of marker and number of repeat to copy prefix to trainval recursively
	# for i in marker #:
	# 	for j in repeat #:
# create trainval, train, val
# copy valid items to corresponding directories (duplicated annotation and images)
# rename object name in annotation files


raw_path = "./"
dst_path = "./marker_raw_data/"
marker_num = 1		# number of marker type
dataset_dup_num = 3 # number of repeated dataset


# # Clear destination directories 
if not(os.path.exists(dst_path)):
	os.makedirs(dst_path)
	os.makedirs(dst_path + "JPEGImages")
	os.makedirs(dst_path + "ImageSets")
	os.makedirs(dst_path + "Annotations")


# append all annotation files, without extension
annotation_xml_list = []
annotation_dirs = os.listdir(raw_path + "Annotations") #n02802426, ...
for path in annotation_dirs:
	xml_list = os.listdir(''.join([raw_path,"Annotations/",path]))
	#remove extension
	for i in xrange(len(xml_list)):
		xml_list[i] = xml_list[i][:-4]
	# append xml_list to annotation_xml_list
	annotation_xml_list += xml_list

# append all image files, without extension
ttl_img_list = []
img_dirs = os.listdir(raw_path + "Images") #n02802426, ...
for path in img_dirs:
	img_list = os.listdir(''.join([raw_path,"Images/",path]))
	#remove extension
	for i in xrange(len(img_list)):
		img_list[i] = img_list[i][:-5]
	# append xml_list to annotation_xml_list
	ttl_img_list += img_list

# get valid dataset (prefix)
valid_prefix = set(ttl_img_list).intersection(annotation_xml_list)
trainval = list(valid_prefix)


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
			src = ''.join([raw_path, "Images/", i[:9], "/", i, ".JPEG"])
			extension = "_marker_" + str(m) + "_" + str(n)
			dst = dst_dir + i + extension + ".JPEG"
			copyfile(src, dst)
			print cnt, '/', len(trainval)*m*n
			cnt += 1


# Copy and modify valid xml
# dst_dir: destination directory
cnt = 1
dst_dir = ''.join([dst_path , "Annotations/"])
for prefix in trainval:
	# for m in xrange(marker_num):
	# 	for n in xrange(dataset_dup_num): 
			# Open original file
			xml_src = ''.join([raw_path, "Annotations/", prefix[:9], "/", prefix, ".xml"])
			tree = ET.parse(xml_src)
			# root = tree.getroot()
			# # Update xml
			# # update filename
			# extension = "_marker_" + str(m) + '_' + str(n)
			# root.find('filename').text = ''.join([prefix, extension])
			# # update name in objects
			# objs = root.findall('object')
			# for obj in objs:
			# 	obj.find('name').text = ''.join(["marker_", str(m)])
			# # Write back to file
			# extension = "_marker_" + str(m) + '_' + str(n)
			# dst = dst_dir + prefix + extension + '.xml'
			dst = dst_dir + prefix + ".xml"
			tree.write(dst)
			print cnt, '/', len(trainval)*marker_num*dataset_dup_num
			cnt += 1