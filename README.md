### Summary


This repository contains an universal marker detection algorithm. The algorithm is developed based on [Faster R-CNN](http://arxiv.org/abs/1506.01497) which was published in NIPS 2015. Given a single marker image, the algorithm is able to train a marker detector which is robust to various cases, including different marker transformation (rotation, occlusion, leaning) and also change in brightness of environment. 

The basic idea behind this algorithm is that, there are thousands of labeled images, which are used for detection training. Each sample set includes an image (.JPEG), an annotation file (.xml) recorded the type and bounding box of objects in the image. These labeled images are used as **raw data** for marker dataset. The original objects inside bounding boxes are replaced by the marker image. Moreover, these modified images are copied to various versions (transformed marker + various brightness). At the end, this dataset contains more than thousands of images for training.

In the following instruction, it includes the guideline for **training and testing in GPU mode**, application in ROS framework, my experience with various issues, and errors that may be encountered and the corresponding solution. Please note that, if you train a model in computer A (**GPU mode**) but run the application in computer B (**CPU mode**), you may refer to Part 7 if there is a reminder as this one: ***CPU mode issue!***. The general procedure for this case is as following.

Computer A:
1. Install the program (Part 2)
2. Prepare training dataset (Part 3)
3. Training and testing (Part 4,5)

Computer B
1. Install the program (Part 2, 7)
2. Copy the trained model from Computer A to Computer B
3. Application with ROS (Part 6)

### Contents
1. [Requirements](#part-1-requirements)
2. [Faster R-CNN installation](#part-2-faster-r-cnn-installation)
3. [Marker detection: dataset](#part-3-marker-detection-dataset)
4. [Marker detection: training](#part-4-marker-detection-training)
5. [Marker detection: testing](#part-5-marker-detection-testing)
6. [Marker detection: application with ROS](#part-6-marker-detection-application-with-ros)
7. [Marker detection: CPU mode issues](#part-7-marker-detection-cpu-mode-issues)
8. [Marker detection: experience](#part-8-marker-detection-experience)
9. [Error and solution](#part-9-error-and-solution)

### Part 1. Requirements

For the marker detection algorithm introduced in this repo, ZF-net is adopted. For **training** smaller networks (ZF-Net), a good GPU (e.g., Titan, K20, K40, ...) with at least **3GB** of memory suffices. The installation of GPU is not included in this post. Therefore, suppose that you have completed the installation of GPU. However, a good GPU is not required in testing/application phase. In other words, if you are going to use the trained model in CPU mode, it is still possible, given that there is enough memory.

### Part 2. Faster R-CNN installation

For the details of Faster R-CNN installation, you may wish to visit my [Caffe installation](https://huangying-zhan.github.io/2016/09/09/GPU-and-Caffe-installation-in-Ubuntu.html#Caffe%20installation) post and [Faster R-CNN](https://huangying-zhan.github.io/2016/09/22/detection-faster-rcnn.html) post. If they are not easy to read, you may also wish to visit the [official instruction](https://github.com/rbgirshick/py-faster-rcnn). In the following parts, I will always refer to my posts for simplicity.

1. Clone the Faster R-CNN repo

    ```Shell
    # Make sure to clone with --recursive
    git clone --recursive https://github.com/Huangying-Zhan/marker-detection.git
    ```

2. Lets call the directory as `$FRCN`. For the following parts, please change this `$FRCN` to your real directory.

3. Build the Cython modules
	
    ***CPU mode issue!***

    ```Shell
    cd $FRCN/lib
    make
    ```

4. Build Caffe and PyCaffe

    For this part, please refer to [Caffe official installation instruction](http://caffe.berkeleyvision.org/installation.html) or my post about [Caffe installation](https://huangying-zhan.github.io/2016/09/09/GPU-and-Caffe-installation-in-Ubuntu.html#Caffe%20installation). 
	If you have experience with Caffe or installed Caffe previously, just follow the instruction below.
    
    ***CPU mode issue!***

    ```Shell
	cd $FRCN/caffe-fast-rcnn
    cp Makefile.config.example Makefile.config

    # Modify Makefile.config, UNCOMMMENT this line
    WITH_PYTHON_LAYER := 1
    # Modifiy Makefile.config according to your need, such as setup related to CPU mode, CUDA version, Anaconda, OpenCV, etc. Basically is to update or enable the functions.

    # After modification on Makefile.config
    make -j4 # 4 is the number of core in your CPU, change it according to your computer CPU  
    # Suppose you have installed prerequites for PyCaffe, otherwise, go back to the Caffe installation instructions.
    make pycaffe -j4
    ```

5. Download pre-computed Faster R-CNN models

	```Shell
    cd $FRCN
	./data/scripts/fetch_faster_rcnn_models.sh
    ```

6. Run the demo

	However, in this part you might get into trouble with different errors, such as without some packages. At the end of this post, some encountered errors and solution are provided. For those unexpected error, google the error and you should be able to find a solution.
    
	```Shell
    ./tools/demo.py
    ```


### Part 3. Marker detection: dataset

#### 3.1. Raw dataset
As mentioned at the begining, we need a raw dataset to construct a new and larger marker detection dataset. The link for the dataset is [here](https://www.dropbox.com/s/3x2u785ys2eqv3l/marker_raw_data.tar.gz?dl=0).

```Shell
# copy the dataset to correct directory
cp marker_raw_data.tar.gz  $FRCN/data/marker/
tar xzf marker_raw_data.tar.gz

# Now you will see a folder named "marker_raw_data", this folder has the following structure.
|-- marker_raw_data
	|-- Annotations
    	|-- *.xml (information about labels and bounding boxes)
    |-- JPEGImages
    	|-- *.JPEG (raw images)
    |-- ImageSets
    	|-- trainval.txt (prefix of annotaion files and image files)
```

#### 3.2. Marker image

Now, you have to prepare your marker image. Actually, this program supports multiple markers. You just need to put your images in the corresponding folder. Here is an example of marker image.

![marker_example](https://cloud.githubusercontent.com/assets/13748721/19136767/950050aa-8ba1-11e6-9aff-dcbf8d421cb3.png)

```Shell
# Suppose the image is called "marker_0.png"
cp marker_0.png $FRCN/data/marker/marker_img/
```

Make sure you **don't put irrelevant files in the folder**! The program automatically trace the files in the folder to build marker dataset later on.

#### 3.3. Marker dataset

Now, we can build our marker dataset based on the raw dataset downloaded in Part 3.1. Personally, I consider this part is the most important part of the whole program. The variety, including both quality and quantity, of the dataset determines the performance of final marker detector. The idea is that, suppose your dataset consists only a single brightness, the detector may not be able to detect markers in darker or brighter environments. Therefore, the construction of the marker dataset should be considered in depth. If you are just using it as a tool, you can use the following script and jump to Part 4.

![dataset_construction_example](https://cloud.githubusercontent.com/assets/13748721/19136792/d39f88f8-8ba1-11e6-8fc1-5471c6328f2b.png)

```Shell
cd $FRCN

# construct marker dataset (default: repeatly copy raw data for 5 times)
./tools/my_tools/dataset_construction.py

# Replace objects in the dataset by marker images
./tools/my_tools/marker_replacement.py
```

In here, I will explain more details about the dataset construction procedures. The dataset construction is basically divided into two-steps. Details can be refered back to the source codes.

##### Step 1. dataset_construction.py

Suppose there are **2** images in the raw dataset. Basically, we need to copy the raw dataset according to our needs. Let's say we have **2** types of markers to be detected and we want to have **4 images for each marker**. We need to copy the raw dataset for **twice** for each marker. Then,

```
|-- raw_dataset
	|-- JPEGImages
        |-- img_0.JPEG
        |-- img_1.JPEG

# The final dataset will have the following format,
|-- marker_dataset
	|-- JPEGImages
		|-- img_0_marker_0_0.JPEG
        |-- img_0_marker_0_1.JPEG
        |-- img_1_marker_0_0.JPEG
        |-- img_1_marker_0_1.JPEG
        |-- img_0_marker_1_0.JPEG
        |-- img_0_marker_1_1.JPEG
        |-- img_1_marker_1_0.JPEG
        |-- img_1_marker_1_1.JPEG
```

Keep in mind that, `img_0_*.JPEG` are totally the same while `img_0_*.JPEG` are the same at this stage. Besides the image files, we also need to construct an annotation file for each new image. The idea is simple: read the original *.xml*　file and then modify the original file to have new annotation files.

```
|-- raw_dataset
	|-- JPEGImages
        |-- img_0.JPEG
        |-- img_1.JPEG
    |-- Annotations
    	|-- img_0.xml
        |-- img_1.xml

# The final dataset will have the following format,
|-- marker_dataset
	|-- JPEGImages
		|-- img_0_marker_0_0.JPEG
        |-- img_0_marker_0_1.JPEG
        |-- img_1_marker_0_0.JPEG
        |-- img_1_marker_0_1.JPEG
        |-- img_0_marker_1_0.JPEG
        |-- img_0_marker_1_1.JPEG
        |-- img_1_marker_1_0.JPEG
        |-- img_1_marker_1_1.JPEG
	|-- Annotations
		|-- img_0_marker_0_0.xml
        |-- img_0_marker_0_1.xml
        |-- img_1_marker_0_0.xml
        |-- img_1_marker_0_1.xml
        |-- img_0_marker_1_0.xml
        |-- img_0_marker_1_1.xml
        |-- img_1_marker_1_0.xml
        |-- img_1_marker_1_1.xml
```

Now, after constructing the dataset, we need to use a *.txt* to save the prefix (i.e. without .JPEG and .xml) of the files such that the program can trace back the files in training phase. Moreover, I have separated the dataset into training set (80%) and validation set (20%).

```
|-- marker_dataset
	|-- JPEGImages
		|-- img_0_marker_0_0.JPEG
        |-- img_0_marker_0_1.JPEG
        |-- img_1_marker_0_0.JPEG
        |-- img_1_marker_0_1.JPEG
        |-- img_0_marker_1_0.JPEG
        |-- img_0_marker_1_1.JPEG
        |-- img_1_marker_1_0.JPEG
        |-- img_1_marker_1_1.JPEG
	|-- Annotations
		|-- img_0_marker_0_0.xml
        |-- img_0_marker_0_1.xml
        |-- img_1_marker_0_0.xml
        |-- img_1_marker_0_1.xml
        |-- img_0_marker_1_0.xml
        |-- img_0_marker_1_1.xml
        |-- img_1_marker_1_0.xml
        |-- img_1_marker_1_1.xml
    |-- ImageSets
    	|-- trainval.txt
        	|-- img_0_marker_0_0
            |-- img_0_marker_0_1
            |-- ...
            }-- img_1_marker_1_1
    	|-- train.txt
        	|-- img_0_marker_0_0
            |-- img_0_marker_0_1
            |-- ...
            }-- img_0_marker_1_1
        |-- val.txt
        	|-- img_1_marker_1_0
            |-- img_1_marker_1_1
        
```

The whole dataset structure is shown above. The details of operations can be refered to the source code.

##### Step 2. marker_replacement.py

As the name given, this step is to replace the originals object inside the bounding box by our marker image. The basic idea/pipeline of the process is shown below. However, the process only describes single marker case. For multiple markers, the idea is just the same but repeating the process with other markers.

```
For each image (Img):
	Get valid bounding box regions (R) in Img. [Annotation files provide the information]
	For each region (rgn) in R:
		1. Transform marker image (M) randomly (rotation, shear, scaling, flip)
        2. Call transformed marker image as M'.
        3. Get size (including height and width) of rgn.
        4. Obtain new marker image (M'') by reshaping M' into the size of rgn.
        5. Replace rgn by M''.
    Randomly change the brightness of the modified Img.
    Save the modified Img
```

Again, the details of the operations can be refered to the source code.

### Part 4. Marker detection: training

In this part, the training of marker detector will be introduced. Again, if you are just using it as a tool, you can just follow the code below and proceed to next part. After executing these commands, the program will start training of a marker detector. At the end of training, it is expected to have some well-trained models (since we will take snapshots of the model from time to time). However, we just need one of them. The final model should be good enough, assumed that it doesn't overfit the training set. Details can be refer to Part 7.

***CPU mode issue!***

```Shell
cd $FRCN

# A part of this bash script is hardcoded. There will be potential problems after modifying the files,train.prototxt, train_init.prototxt and test.prototxt at $FRCN/models/marker/
# Usage:
# ./experiments/scripts/marker_detection.sh [GPU_ID] [Training_Iteration]
# For GPU_ID, you may check by this command: nvidia-smi. Normally it is 0.
./experiments/scripts/marker_detection.sh 0 50000
```

At the end , you will get some trained models at this directory, `$FRCN/output/marker/`. We just need one of them, let's use the final one. Now you can jump to Part 5 if you don't need to know more about the training. 

If you wish to know more about training in Caffe, here provides more details.
Basically, we need to prepare a prototxt (defining network structure), a prototxt defining hyper-paremeters (e.g. learning rate, learning policy), a configuration file (config.xml) and a pre-trained model (.caffemodel) for network parameter initialization. For the details of whole workflow of py-faster-rcnn, you may wish to visit my [Detection: Faster R-CNN](https://huangying-zhan.github.io/2016/09/22/detection-faster-rcnn.html) post to know more details. The post includes the workflow behind py-faster-rcnn and an example of basketball detection. Actually the idea behind the example is similar to marker detection.

### Part 5. Marker detection: testing

In this part, we will conduct a testing on our trained model. As mentioned before, we have 80% of the dataset as training set while remaining as validation set. In this testing, we can use the validation set to test the performance of our trained model. The performance is reflected by mAP (mean Average Precision). However, our validation set is **not a natural** dataset since the marker images are digitally-altered (photoshopped). The validation set might not indicate the actual performance.

In here, we will introduce two testing methods. First one is using validation set; Second one is using new test images and observe the performance manually. Second one is recommended.

#### 5.1. Validation set

Run the following command to check the performance.

```Shell
# Suppose your trained model from Part 4 is called zf_faster_rcnn_marker_iter_50000.caffemodel
./tools/test.py --gpu 0 --def models/marker/test.prototxt　--net output/marker/train/zf_faster_rcnn_marker_iter_50000.caffemodel --imdb marker_val cfgexperiments/cfgs/config.yml
```

#### 5.2. Test images

***CPU mode issue!***

```Shell
# Put test images in demo diretory at $FRCN/data/demo
# Update the caffemodel path @line 118 @$FRCN/tools/marker_demo.py if necessary
./tools/marker_demo.py
```

### Part 6. Marker detection: application with ROS

In this part, we will see how to use a trained model for marker detection in application. Basically, you can treat it as a package which equiped with a trained model. If you provide a RGB image, the package is able to return the position of the marker in the image if there exists.

Suppose you have installed ROS and trained a model for marker detection.
Put the model at `$FRCN/output/marker/train/ros/`.

***CPU mode issue!***

```Shell
# Before creating ROS package, it is suggested to install the following Python packages first since the marker detection ROS package is developed in Python.
pip install empy
pip install catkin_pkg
pip install rosdep

# Create a new ROS workspace for this marker detection. 
cd $FRCN
./tools/my_tools/ros_package/ros.sh

# Run ROS
cd $FRCN/catkin_ws
roscore

# Open a new terminal and run marker detection ROS package
# This ROS package read in an image and output detection result
source devel/setup.bash
rosrun marker_detection marker_detection_ros.py

# For debugging, an external image publisher and result subscriber are also implemented.
# Open a new terminal
source devel/setup.bash
rosrun marker_detection external_image.py
# Open a new terminal
source devel/setup.bash
rosrun marker_detection external_result.py
```

#### `marker_detection_ros.py`

##### Function

`marker_detection_ros.py` subscribes an image topic message and perform detection after receiving the message. After detection, it publishes a topic message regarding the detection result. Moreover, for each received image, the resulted image is saved at `$FRCN/catkin_ws/src/marker_detection/detected_img`. This folder will be **cleared for new images** when new detection task begins.

##### Topic messages

Concerning the topic messages subscribed and published by the marker_detection_ros.py, the messages have the following format.

```Python
# As an image subscriber, it subscribes the standard Image format in sensro_msgs.msg.
# As a detection result publisher, it releases 3 types of detection result
- bool marker_detected # tells whether a marker is detected or not
- float32[] prob  	 # tells the confidence with the detected result, ranging from 0 to 1
- bbox[] bboxes		 # tells the coordinate of top-left corner and bottom-right corner of detected bounding box

# bbox[] is a self-defined message. It has the following message format.
- int32[4] bbox # The 4 integers indicate [row,col] of top-left corner and [row,col] of bottom-right corner. 

```


### Part 7. Marker detection: CPU mode issues

##### Part 2.3. Build the Cython modules

You may find `EnvironmentError: The nvcc binary could not be located in your $PATH` if CUDA is not installed well or GPU mode is not available in your computer.

This error is caused by lack of CUDA. If you have an appropriate NVidia GPU device, you may wish to install CUDA first (refer to [here](https://huangying-zhan.github.io/2016/09/09/GPU-and-Caffe-installation-in-Ubuntu.html#title4)). Othervise, you should annotate the code related to GPU in `$FRCN/lib/setup.py`.

```
...
#CUDA = locate_cuda()
...
...
#self.set_executable('compiler_so', CUDA['nvcc'])
...
...
#Extension('nms.gpu_nms',
# ['nms/nms_kernel.cu', 'nms/gpu_nms.pyx'],
# library_dirs=[CUDA['lib64']],
# libraries=['cudart'],
# language='c++',
# runtime_library_dirs=[CUDA['lib64']],
# # this syntax is specific to this build system
# # we're only going to use certain compiler args with nvcc and not with
# # gcc the implementation of this trick is in customize_compiler() below
# extra_compile_args={'gcc': ["-Wno-unused-function"],
# 'nvcc': ['-arch=sm_35',
# '--ptxas-options=-v',
# '-c',
# '--compiler-options',
# "'-fPIC'"]},
# include_dirs = [numpy_include, CUDA['include']]
# )
```

##### Part 2.4. Build Caffe and PyCaffe

This part specifically focus on application phase in CPU mode. Suppose you have already trained a model for implementation. The following part focus on application phase in CPU mode.

```Shell
cd $FRCN/caffe-fast-rcnn
cp Makefile.config.example Makefile.config

# Modify Makefile.config
# Enable python layers, uncomment this line
WITH_PYTHON_LAYER := 1
# Modifiy Makefile.config according to your need, such as setup related to CPU mode, CUDA version, Anaconda, OpenCV, etc.
# Enable CPU mode, uncomment this line
CPU_ONLY:=1
# Enable Anaconda, uncomment the following lines. Update ANACONDA_HOME if necessary.
ANACONDA_HOME := $(HOME)/anaconda 
PYTHON_INCLUDE := $(ANACONDA_HOME)/include \
		 $(ANACONDA_HOME)/include/python2.7 \
		 $(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
# After modification on Makefile.config
make -j4 # 4 is the number of core in your CPU, change it according to your computer CPU  
# Suppose you have installed prerequites for PyCaffe, otherwise, go back to the Caffe installation instructions.
make pycaffe -j4
```

##### Part 4. Marker detection: training

Basically, training is not operated in CPU mode. However, if the ultimate goal for your project is to implement the algorithm in a computer without GPU, there is a trick to improve the detection speed. The trick is to reduce image size in testing/implementation phase. However, this will reduce your detection performance (precision). The reason is that you train the model with large images while testing with relatively small images. To improve the performance, we can train the network model with small images. The trick is to modify `$FRCN/lib/fast_rcnn/config.py`. Besides image size, we can also alter the number of bounding boxes after applying non-maxima suppression to RPN proposals. When number of proposals is reduced, the detection speed is also improved.

```Python
# Change the scale of image
# For training phase,
__C.TRAIN.SCALES = (600,)
__C.TRAIN.MAX_SIZE = 1000
# These two values limit the minimum size of an image's shortest side and longest side's maximum size. 
# The minimum suggested values for this marker detection algorithm in CPU mode is that,
__C.TRAIN.SCALES = (200,)
__C.TRAIN.MAX_SIZE = 400
# However, we should also update the scale in testing phase.
__C.TEST.SCALES = (200,)
__C.TEST.MAX_SIZE = 400

# Reduce number of bounding boxes in testing phase
# This value indicates the number of proposals you are going to classify as marker or not at the end. The less the faster.
__C.TEST.RPN_POST_NMS_TOP_N = 300
```

##### Part 5.2. Test images

To run `$FRCN/tools/marker_demo.py` in CPU mode. We should use this command in order to enable CPU mode.

```Shell
./tools/marker_demo.py --cpu
```


##### Part 6. Marter detection: application with ROS

If you have modified `$FRCN/lib/fast_rcnn/config.py` for smaller images, and this application with ROS is running in another computer different from the computer for training, you also need to update `$FRCN/lib/fast_rcnn/config.py` since the scripts in this part also call this `config.py`.

```Python
# Basically, you only need to update this part and make it as same as training setup.
__C.TEST.SCALES = (200,)
__C.TEST.MAX_SIZE = 400
```

Moreover, there are some files need to be updated to disable GPU related functions.

```Python
# $FRCN/lib/fast_rcnn/config.py
__C.USE_GPU_NMS = True -> __C.USE_GPU_NMS = False
# $FRCN/lib/fast_rcnn/nms_wrapper.py, comment the following line
# from nms.gpu_nms import gpu_nms
# $FRCN/tools/test_net.py
caffe.set_mode_gpu() -> caffe.set_mode_cpu()

```

### Part 8. Marker detection: experience

In this part, I would like to share some of my experience in this marker detection project. There are three areas I am going to discuss, including dataset construction, training, and evaluation.

#### 8.1. Dataset construction

Currently, this program is using an unified framework to construct dataset and training. The overal performance is pretty robust. However, this final framework is not completed at once. I did some experiments and finally came out with this framework. In the following paragraphs, I would like to share the experience and observation concerning this dataset construction. All are based on empirical experience. There is not yet any theorectical explanation. 

##### Marker image transformation

![dataset_construction_example](https://cloud.githubusercontent.com/assets/13748721/19136792/d39f88f8-8ba1-11e6-8fc1-5471c6328f2b.png)

At the begining of the whole project, there is only one marker image. I just resize the marker image according to size of bounding boxes and directly replace the region by the marker image. In other words, the images in whole dataset only used same marker image inside object regions.

However, a problem is revealed at testing phase. The following photos are not able to be found out by the detector even they are clearly recognized by human beings.

From this observation, the possible reason might be that the dataset only consists single type of the marker image. The variety is not enough to train a detector to handle various cases. Therefore, a data augmentation algorithm is introduced to handle such cases. The image augmentor is able to transform an image with various options (all are random), including,

* Verticle flip (flip with certain %)
* Horizontal flip (flip with certain %)
* Shear (random angle in a certain range)
* Rotation (random angle in a certain range)
* Scaling (random scaling with a scaling range)

After applying these random transformation on marker image before pasting on object regions, the newly created dataset should be able to handle various cases, including the previous failed images. As I just mentioned, there are various options which can be adjusted. Although the current setting should be able handle various cases. You may come back to this part if there is any specific environment or requirement on your marker image.

##### Brightness enhancement

However, there is another new issue I encoutered after transformation, which are images with different brightness. The various brightness might be caused by camera exporsure time or the environment itself. However, here are some failed examples. 

![failed_example](https://cloud.githubusercontent.com/assets/13748721/19136893/88b58efe-8ba2-11e6-8ade-8f62828cb981.png)

The first possible solution I came out is similar to marker image transformation. Tune the brightness of marker images and put the new markers on original object regions. This method sometimes can handle this issue but the result is unstable. Generally, the image should has similar brightness anywhere in the image. The whole brightness should be smooth rather than sudden change in certain regions (marker image region). After that, I guessed that the reason might not be the marker image itself but the whole image. The detector is not able to find out the marker from an image/environment with unseen brightness rather than the detector is not able to recognize the marker with various brightness. Therefore, the brightness of whole image of all images in the dataset is tuned randomly rather than tuning the brightness of marker itself only. This idea finally made the detector becoming very robust in various cases.

##### Dataset size

Dataset size should be another important issue in this algorithm. Originally, I just used the raw dataset with around 3000 images for marker replacement and training. After marker image transformation and brightness enhancement. The performance is not that robust yet. I guessed that it might be caused by the size of dataset since we altered marker image with many random pre-processing. The dataset might not be capable to include most of the random pre-processing.
Therefore, I enlarge the dataset by duplicating original dataset. Finally, this approach improves the performance.

#### 8.2. Training and testing tricks

In this part, I would like to share some experience concerning test result observation. One of the drawback of this algorithm is that there is not **natural** validation dataset to evaluate the performance of the detector by a systematic and numerical way, given that you don't provide a manually labelled validation set. Therefore, basically the performance can be only evaluated by observation on some test images without annotation.

By observation on test result, if we observed that the detector is only able to response to pattherns in training set but unable to detect new patterns, this is probably an overfitting problem. Basically it is caused by inadequate dataset (lack of variety). We can enlarge the dataset to solve this issue.

Another possible observation is that, you have enlarged your dataset but you found that the detector doesn't have very good performance. For example, it is able to detect marker but inaccurate bounding box, or some false positives. You can train the network with more iterations allowing the detector to increase inter-class (marker & background) distance and converge better.

While using this program for applications, there is an important parameter in testing/real application. That is something called CONF_THRESH. It stands for confidence threshold. Simply speaking, it is a threshold to reject bounding boxes with probability less than it. However, sometimes our detector is able to detect marker but with lower probability (e.g. ~50%) due to environment conditions. In such cases, we may try to tune the CONF_THRESH to a lower value such that more positives are accepted. You may concern the false positive cases. From previous experience, the detector is quite robust to reject non-marker proposals, especially in clean image (e.g. a marker placed on a meadow). In other words, the detector only response to marker.


### Part 9. Error and solution


1. no easydict, cv2

    ```Shell
    # Without Anaconda
    sudo pip install easydict
    sudo apt-get install python-opencv
    
    # With Anaconda
    conda install opencv
    conda install -c verydeep easydict
    # Normally, people will follow the online instruction at https://anaconda.org/auto/easydict and install auto/easydict. However, this easydict (ver.1.4) has a problem in passing the message of configuration and cause many unexpected error while verydeep/easydict (ver.1.6) won't cause these errors.
    ```

2. libcudart.so.8.0: cannot open shared object file: No such file or directory
	
    ```Shell
    sudo ldconfig /usr/local/cuda/lib64
    # if this doesn't work, then
    export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:$LD_LIBRARY_PATH 
    # Permanent solution: add this export command to ~/.bashrc
    ```

3. assertionError: Selective Search data is not found

	Solution: install verydeep/easydict rather than auto/easydict
    
    ```Shell
    conda install -c verydeep easydict
    ```

4. box [:, 0] > box[:, 2]

	Solution: add the following code block in imdb.py
    
    ```Python
    def append_flipped_images(self):
        num_images = self.num_images
        widths = self._get_widths()
        for i in xrange(num_images):
            boxes = self.roidb[i]['boxes'].copy()
            oldx1 = boxes[:, 0].copy()
            oldx2 = boxes[:, 2].copy()
            boxes[:, 0] = widths[i] - oldx2
            boxes[:, 2] = widths[i] - oldx1
            for b in range(len(boxes)):
                    if boxes[b][2] < boxes[b][0]:
                        boxes[b][0]=0
            assert (boxes[:, 2] >= boxes[:, 0]).all()
    ```

5. For ImageNet detection dataset, no need to minus *one* on coordinates

	```Python
    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        cls = self._class_to_ind[obj.find('name').text.lower().strip()]
	```
    
6. No module named Cython.Distuils
	
    Install Anaconda can solve problem. Remember to re-start a new terminal after installation of Anaconda.
   
7. No module named Cython.Distuils

	If it appears "No module named Cython.Distuils", it is recommended that you should install Anaconda first. Please refer to [here](https://huangying-zhan.github.io/2016/09/09/GPU-and-Caffe-installation-in-Ubuntu.html#title6).

8. No module named gpu_nms

	```Python
    # $FRCN/lib/faster_rcnn/config.py
    __C.USE_GPU_NMS = True -> __C.USE_GPU_NMS = False
    # $FRCN/tools/test_net.py
    caffe.set_mode_gpu() -> caffe.set_mode_cpu()
    # $FRCN/lib/fast_rcnn/nms_wrapper.py, comment the following line
    # from nms.gpu_nms import gpu_nms
    ```
    
9. Python ROS: No module named xxx
    
	```Shell
    # No module named rospkg
    git clone git://github.com/ros/rospkg.git 
    cd rospkg
    python setup.py install --user
    
    # No module named em
    pip install empy
    
    # No module named catkin_pkg
	pip install catkin_pkg

	# No module named rosdep2.rospack
	pip install rosdep
    ```
    
11. ImprotError: libglog.so.0: cannot open shared object file: No such file or directory

    ```Shell
    # Temporary solution
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
    # Permanent solution: add this line to ~/.bashrc
    ```
