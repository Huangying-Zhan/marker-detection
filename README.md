### Summary

To be completed
5.2
part 6
imgs


This repository contains an universal marker detection algorithm. The algorithm is developed based on [Faster R-CNN](http://arxiv.org/abs/1506.01497) which was published in NIPS 2015. Given a single marker image, the algorithm is able to train a marker detector which is robust to various cases, including different marker transformation (rotation, occlusion, leaning) and also change in brightness of environment. 

The basic idea behind this algorithm is that, there are thousands of labeled images, which are used for detection training. Each sample set includes an image (.JPEG), an annotation file (.xml) recorded the type and bounding box of objects in the image. These labeled images are used as **raw data** for marker dataset. The original objects inside bounding boxes are replaced by the marker image. Moreover, these modified images are copied to various versions (transformed marker + various brightness). At the end, this dataset contains more than thousands of images for training.

### Contents
1. [Requirements](#part-1-requirements)
2. [Faster R-CNN installation](#part-2-faster-r-cnn-installation)
3. [Marker detection: dataset](#part-3-marker-detection-dataset)
4. [Marker detection: training](#part-4-marker-detection-training)
5. [Marker detection: testing](#part-5-marker-detection-testing)
6. [Marker detection: application](#part-6-marker-detection-application)
7. [Marker detection: experience](#part-7-marker-detection-experience)
8. [Error and solution](#part-8-error-and-solution)

### Part 1. Requirements

For the marker detection algorithm introduced in this repo, ZF-net is adopted. For training smaller networks (ZF-Net), a good GPU (e.g., Titan, K20, K40, ...) with at least **3GB** of memory suffices. The installation of GPU is not included in this post. Therefore, suppose that you have completed the installation of GPU. 

### Part 2. Faster R-CNN installation

For the details of Faster R-CNN installation, you may wish to visit my [Caffe installation](https://huangying-zhan.github.io/2016/09/09/GPU-and-Caffe-installation-in-Ubuntu.html#Caffe%20installation) post and [Faster R-CNN](https://huangying-zhan.github.io/2016/09/22/detection-faster-rcnn.html) post. If they are not easy to read, you may also wish to visit the [official instruction](https://github.com/rbgirshick/py-faster-rcnn). In the following parts, I will always refer to my posts for simplicity. 

1. Clone the Faster R-CNN repo

  ```
  # Make sure to clone with --recursive
  git clone --recursive git@bitbucket.org:JZ3627/marker-detection.git
  ```

2. Lets call the directory as `$FRCN`. For the following parts, please change this `$FRCN` to your real directory.

3. Build the Cython modules

    ```
    cd $FRCN/lib
    make
    ```
    
4. Build Caffe and PyCaffe
	
    For this part, please refer to [Caffe official installation instruction](http://caffe.berkeleyvision.org/installation.html) or my post about [Caffe installation](https://huangying-zhan.github.io/2016/09/09/GPU-and-Caffe-installation-in-Ubuntu.html#Caffe%20installation). 
	If you have experience with Caffe or installed Caffe previously, just follow the instruction below.
    
    ```
	cd $FRCN/caffe-fast-rcnn
    cp Makefile.config.example Makefile.config
    
    # Modify Makefile.config, uncommment this line
    WITH_PYTHON_LAYER := 1
    # Modifiy Makefile.config according to your need, such as setup related to GPU support, cuDNN, CUDA version, Anaconda, OpenCV, etc.
    
    # After modification on Makefile.config
    make -j4 # 4 is the number of core in your CPU, change it according to your computer CPU  
    # Suppose you have installed prerequites for PyCaffe, otherwise, go back to the Caffe installation instructions.
    make pycaffe -j4
    ```
    
5. Download pre-computed Faster R-CNN models

	```
    cd $FRCN
	./data/scripts/fetch_faster_rcnn_models.sh
    ```

6. Run the demo

	However, in this part you might get into trouble with different errors, such as without some packages. At the end of this post, some encountered errors and solution are provided. For those unexpected error, google the error and you should be able to find a solution.
    
	```
    ./tools/demo.py
    ```


### Part 3. Marker detection: dataset

#### 3.1. Raw dataset
As mentioned at the begining, we need a raw dataset to construct a new and larger marker detection dataset. The link for the dataset is [here](https://www.dropbox.com/s/3x2u785ys2eqv3l/marker_raw_data.tar.gz?dl=0).

```
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

![img](marker image example)

```
# Suppose the image is called "marker_0.png"
cp marker_0.png $FRCN/data/marker/marker_img/
```

Make sure you **don't put irrelevant files in the folder**!

#### 3.3. Marker dataset

Now, we can build our marker dataset based on the raw dataset downloaded in Part 3.1. Personally, I consider this part is the most important part of the whole program. The variety, including both quality and quantity, of the dataset determines the performance of final marker detector. The idea is that, suppose your dataset consists only a single brightness, the detector may not be able to detect markers in darker or brighter environments. Therefore, the construction of the marker dataset should be considered in depth. If you are just using it as a tool, you can use the following script and jump to Part 4.

![img](duplicate example, 1 to 5 to modified 5)

```
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

Keep in mind that, `img_0_*.JPEG` are totally the same while `img_0_*.JPEG` are same at this stage. Besides the image files, we also need to construct an annotation file for each new image. The idea is simple, read in the original *.xml*　file and then modify the original file to have new annotation files.

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

```
cd $FRCN

# A part of this bash script is hardcoded. There will be potential problems after modifying the file, models/marker/train.prototxt 
# Usage:
# ./experiments/scripts/marker_detection.sh [GPU_ID] [Training_Iteration]
# For GPU_ID, you may check by this command: nvidia-smi. Normally it is 0.
./experiments/scripts/marker_detection.sh 0 50000
```

At the end , you will get some trained models at this directory, `$FRCN/output/marker/`. We just need one of them, let's use the final one. Now you can jump to Part 5 if you don't need to know more about the training.

Basically, we need to prepare a prototxt (defining network structure), a prototxt defining hyper-paremeters (e.g. learning rate, learning policy), a configuration file (config.xml) and a pre-trained model (.caffemodel) for network parameter initialization. For the details of whole workflow of py-faster-rcnn, you may wish to visit my [Detection: Faster R-CNN](https://huangying-zhan.github.io/2016/09/22/detection-faster-rcnn.html) post to know more details. The post includes the workflow behind py-faster-rcnn and an example of basketball detection. Actually the idea behind the example is similar to marker detection.


### Part 5. Marker detection: testing

In this part, we will conduct a testing on our trained model. As mentioned before, we have 80% of the dataset as training set while remaining as validation set. In this testing, we can use the validation set to test the performance of our trained model. The performance is reflected by mAP (mean Average Precision). However, our validation set is **not a natural** dataset since the marker images are digitally-altered (photoshopped). The validation set might not indicate the actual performance.

In here, we will introduce two testing methods. First one is using validation set; Second one is using new test images and observe the performance manually. Second one is recommended.

#### 5.1. Validation set

Run the following command to check the performance.
```
# Suppose your trained model from Part 4 is called zf_faster_rcnn_marker_iter_50000.caffemodel
./tools/test.py --gpu 0 --def models/marker/test.prototxt　--net output/marker/train/zf_faster_rcnn_marker_iter_50000.caffemodel --imdb marker_val cfgexperiments/cfgs/config.yml
```

#### 5.2. Test images

```
put images in demo diretory
./tools/marker_detection.py
```

### Part 6. Marker detection: application

In this part, we will see how to use a trained model for marker detection in application. Basically, you can treat it as a package which equiped with a trained model. If you provide a RGB image, the package is able to return the position of the marker if there exists.

```
setup network and model
setup detection.py
return values
```

### Part 7. Marker detection: experience

In this part, I would like to share some of my experience in this marker detection project. There are three areas I am going to discuss, including dataset construction, training, and evaluation.

#### 7.1. Dataset construction

Currently, this program is using an unified framework to construct dataset and training. The overal performance is pretty robust. However, this final framework is not completed at once. I did some experiments and finally came out with this framework. In the following paragraphs, I would like to share the experience and observation concerning this dataset construction. All are based on empirical experience. There is not yet any theorectical explanation. 

##### Marker image transformation

At the begining of the whole project, there is only one marker image. I just resize the marker image according to size of bounding boxes and directly replace the region by the marker image. In other words, the images in whole dataset only used same marker image inside object regions.

![img](key same marker to some imgs)

However, a problem is revealed at testing phase. The following photos are not able to be found out by the detector even they are clearly recognized by human beings.

![imgs](rotation, shear images)

From this observation, the possible reason might be that the dataset only consists single type of the marker image. The variety is not enough to train a detector to handle various cases. Therefore, a data augmentation algorithm is introduced to handle such cases. The image augmentor is able to transform an image with various options (all are random), including,

* Verticle flip (flip with certain %)
* Horizontal flip (flip with certain %)
* Shear (random angle in a certain range)
* Rotation (random angle in a certain range)
* Scaling (random scaling with a scaling range)

After applying these random transformation on marker image before pasting on object regions, the newly created dataset should be able to handle various cases, including the previous failed images. As I just mentioned, there are various options which can be adjusted. Although the current setting should be able handle various cases. You may come back to this part if there is any specific environment or requirement on your marker image.

##### Brightness enhancement

However, there is another new issue I encoutered after transformation, which are images with different brightness. The various brightness might be caused by camera exporsure time or the environment itself. However, here are some failed examples. 

![imgs](dark and over exposure images)

The first possible solution I came out is similar to marker image transformation. Tune the brightness of marker images and put the new markers on original object regions. This method sometimes can handle this issue but the result is unstable. Generally, the image should has similar brightness anywhere in the image. The whole brightness should be smooth rather than sudden change in certain regions (marker image region). After that, I guessed that the reason might not be the marker image itself but the whole image. The detector is not able to find out the marker from an image/environment with unseen brightness rather than the detector is not able to recognize the marker with various brightness. Therefore, the brightness of whole image of all images in the dataset is tuned randomly rather than tuning the brightness of marker itself only. This idea finally made the detector becoming very robust in various cases.

##### Dataset size

Dataset size should be another important issue in this algorithm. Originally, I just used the raw dataset with around 3000 images for marker replacement and training. After marker image transformation and brightness enhancement. The performance is not that robust yet. I guessed that it might be caused by the size of dataset since we altered marker image with many random pre-processing. The dataset might not be capable to include most of the random pre-processing.
Therefore, I enlarge the dataset by duplicating original dataset. Finally, this approach improves the performance.

#### 7.2. Training and testing tricks

In this part, I would like to share some experience concerning test result observation. One of the drawback of this algorithm is that there is not **natural** validation dataset to evaluate the performance of the detector by a systematic and numerical way, given that you don't provide a manually labelled validation set. Therefore, basically the performance can be only evaluated by observation on some test images without annotation.

By observation on test result, if we observed that the detector is only able to response to pattherns in training set but unable to detect new patterns, this is probably an overfitting problem. Basically it is caused by inadequate dataset (lack of variety). We can enlarge the dataset to solve this issue.

Another possible observation is that, you have enlarged your dataset but you found that the detector doesn't have very good performance. For example, it is able to detect marker but inaccurate bounding box, or some false positives. You can train the network with more iterations allowing the detector to increase inter-class (marker & background) distance and converge better.

While using this program for applications, there is an important parameter in testing/real application. That is something called CONF_THRESH. It stands for confidence threshold. Simply speaking, it is a threshold to reject bounding boxes with probability less than it. However, sometimes our detector is able to detect marker but with lower probability (e.g. ~50%) due to environment conditions. In such cases, we may try to tune the CONF_THRESH to a lower value such that more positives are accepted. You may concern the false positive cases. From previous experience, the detector is quite robust to reject non-marker proposals, especially in clean image (e.g. a marker placed on a meadow). In other words, the detector only response to marker.


### Part 8. Error and solution


1. no easydict, cv2

    ```
    # Without Anaconda
    sudo pip install easydict
    sudo apt-get install python-opencv
    
    # With Anaconda
    conda install opencv
    conda install -c verydeep easydict
    # Normally, people will follow the online instruction at https://anaconda.org/auto/easydict and install auto/easydict. However, this easydict (ver.1.4) has a problem in passing the message of configuration and cause many unexpected error while verydeep/easydict (ver.1.6) won't cause these errors.
    ```

2. libcudart.so.8.0: cannot open shared object file: No such file or directory
	
    	sudo ldconfig /usr/local/cuda/lib64

3. assertionError: Selective Search data is not found

	Solution: install verydeep/easydict rather than auto/easydict
    
    ```
    conda install -c verydeep easydict
    ```

4. box [:, 0] > box[:, 2]

	Solution: add the following code block in imdb.py
    
    ```
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

	```
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