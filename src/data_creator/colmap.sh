IMAGE_PATH=/home/bharathsk/datasets/drop/frames/drop_3
WORKSPACE_PATH=/home/bharathsk/datasets/drop/colmap/drop_3
QUALITY=high
DATA_TYPE=video
SINGLE_CAMERA=1
CAMERA_TYPE=PINHOLE
DENSE=0

mkdir $WORKSPACE_PATH
colmap automatic_reconstructor --image_path $IMAGE_PATH --workspace_path $WORKSPACE_PATH --quality $QUALITY --data_type $DATA_TYPE --single_camera $SINGLE_CAMERA --camera_model $CAMERA_TYPE --dense $DENSE

IMAGE_PATH=/home/bharathsk/datasets/drop/frames/drop_4
WORKSPACE_PATH=/home/bharathsk/datasets/drop/colmap/drop_4
QUALITY=high
DATA_TYPE=video
SINGLE_CAMERA=1
CAMERA_TYPE=PINHOLE
DENSE=0

mkdir $WORKSPACE_PATH
colmap automatic_reconstructor --image_path $IMAGE_PATH --workspace_path $WORKSPACE_PATH --quality $QUALITY --data_type $DATA_TYPE --single_camera $SINGLE_CAMERA --camera_model $CAMERA_TYPE --dense $DENSE

IMAGE_PATH=/home/bharathsk/datasets/drop/frames/drop_5
WORKSPACE_PATH=/home/bharathsk/datasets/drop/colmap/drop_5
QUALITY=high
DATA_TYPE=video
SINGLE_CAMERA=1
CAMERA_TYPE=PINHOLE
DENSE=0

mkdir $WORKSPACE_PATH
colmap automatic_reconstructor --image_path $IMAGE_PATH --workspace_path $WORKSPACE_PATH --quality $QUALITY --data_type $DATA_TYPE --single_camera $SINGLE_CAMERA --camera_model $CAMERA_TYPE --dense $DENSE

IMAGE_PATH=/home/bharathsk/datasets/drop/frames/drop_6
WORKSPACE_PATH=/home/bharathsk/datasets/drop/colmap/drop_6
QUALITY=high
DATA_TYPE=video
SINGLE_CAMERA=1
CAMERA_TYPE=PINHOLE
DENSE=0

mkdir $WORKSPACE_PATH
colmap automatic_reconstructor --image_path $IMAGE_PATH --workspace_path $WORKSPACE_PATH --quality $QUALITY --data_type $DATA_TYPE --single_camera $SINGLE_CAMERA --camera_model $CAMERA_TYPE --dense $DENSE