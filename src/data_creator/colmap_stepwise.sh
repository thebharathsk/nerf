
#!/bin/bash

names=("drop_2" "drop_3" "drop_4" "drop_5" "drop_6")

for name in "${names[@]}"
do
    IMAGES=/home/bharathsk/datasets/drop/frames/$name
    PROJECT=/home/bharathsk/datasets/drop/colmap/$name

    mkdir $PROJECT
    mkdir $PROJECT/sparse
    mkdir $PROJECT/dense

    # colmap feature_extractor\
    # --ImageReader.camera_model PINHOLE \
    # --ImageReader.single_camera 1 \
    # --database_path $PROJECT/database.db\
    # --image_path $IMAGES

    # colmap exhaustive_matcher\
    # --database_path $PROJECT/database.db

    # colmap mapper\
    # --database_path $PROJECT/database.db \
    # --Mapper.ba_refine_extra_params 0 \
    # --image_path $IMAGES \
    # --output_path $PROJECT/sparse 

    colmap image_undistorter\
    --image_path $IMAGES\
    --input_path $PROJECT/sparse/0\
    --output_path $PROJECT/dense

    colmap patch_match_stereo\
    --workspace_path $PROJECT/dense

    colmap stereo_fusion\
    --workspace_path $PROJECT/dense\
    --output_path $PROJECT/dense/fused.ply\
    --input_type geometric

    colmap poisson_mesher\
    --input_path $PROJECT/dense/fused.ply\
    --output_path $PROJECT/dense/meshed-poisson.ply
done