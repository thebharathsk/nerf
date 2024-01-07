IMAGE_PATH=/home/bharathsk/datasets/drop/frames/drop_2
WORKSPACE_PATH=/home/bharathsk/datasets/drop/colmap/drop_2
QUALITY=high
DATA_TYPE=video
SINGLE_CAMERA=1
CAMERA_TYPE=SIMPLE_PINHOLE
DENSE=0

mkdir $WORKSPACE_PATH
colmap automatic_reconstructor --image_path $IMAGE_PATH --workspace_path $WORKSPACE_PATH --quality $QUALITY --data_type $DATA_TYPE --single_camera $SINGLE_CAMERA --camera_model $CAMERA_TYPE --dense $DENSE