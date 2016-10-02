# Read inputs
GPU_ID=$1

# Update solver.prototxt
echo "Updating solver..."
sed -i 's/train.prototxt/train_init.prototxt/' ./models/marker/solver.prototxt

# Get number of marker images
echo "Checking maker images..."
MARKER_NUM=`ls data/marker/marker_img/ | wc -l`
echo "Number of marker is $MARKER_NUM."

# Update train_init.prototxt according to number of marker
echo "Updating train.prototxt and test.prototxt..."
CLS_NUM=$(($MARKER_NUM+1))
BBOX_NUM=$(($CLS_NUM*4))

line="    param_str: \"'num_classes': $CLS_NUM\""
sed -i "11s/.*/$line/" models/marker/train_init.prototxt

line="     param_str: \"'num_classes': $CLS_NUM\""
sed -i "364s/.*/$line/" models/marker/train_init.prototxt

line="    num_output: $CLS_NUM"
sed -i "444s/.*/$line/" models/marker/train_init.prototxt

line="    num_output: $BBOX_NUM"
sed -i "463s/.*/$line/" models/marker/train_init.prototxt

# Update train.prototxt according to number of marker
line="    param_str: \"'num_classes': $CLS_NUM\""
sed -i "11s/.*/$line/" models/marker/train.prototxt

line="     param_str: \"'num_classes': $CLS_NUM\""
sed -i "364s/.*/$line/" models/marker/train.prototxt

line="    num_output: $CLS_NUM"
sed -i "444s/.*/$line/" models/marker/train.prototxt

line="    num_output: $BBOX_NUM"
sed -i "463s/.*/$line/" models/marker/train.prototxt

# Update test.prototxt according to number of marker
line="    num_output: $CLS_NUM"
sed -i "352s/.*/$line/" models/marker/test.prototxt

line="    num_output: $BBOX_NUM"
sed -i "361s/.*/$line/" models/marker/test.prototxt

# Set up a log to store training process
LOG="experiments/logs/marker_detection.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# remove cache
echo "Removing cache..."
rm ./data/cache/*

# First training
echo "Initial training..."
./tools/train_net.py --gpu ${GPU_ID} \
	--weights data/faster_rcnn_models/ZF_faster_rcnn_final.caffemodel \
	--imdb marker_train \
	--cfg experiments/cfgs/config.yml \
	--solver models/marker/solver.prototxt \
	--iters 0

WEIGHT_INIT=`grep -B 1 "done solving" ${LOG} | grep "Wrote snapshot" | awk '{print $4}'`

# Update solver.prototxt
echo "Updating solver..."
sed -i 's/train_init.prototxt/train.prototxt/' ./models/marker/solver.prototxt

# Final training
echo "Start final training..."
./tools/train_net.py --gpu ${GPU_ID} \
	--weights ${WEIGHT_INIT} \
	--imdb marker_train \
	--cfg experiments/cfgs/config.yml \
	--solver models/marker/solver.prototxt \
	--iters 50000