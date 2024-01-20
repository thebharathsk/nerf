PROJECT=/home/bharathsk/datasets/drop/colmap/drop_4/
colmap feature_extractor --SiftExtraction.use_gpu 0 --ImageReader.camera_model PINHOLE --ImageReader.single_camera 1 --database_path $PROJECT/database.db --image_path $PROJECT/images

colmap exhaustive_matcher --SiftMatching.use_gpu 0 --database_path $PROJECT/database.db

colmap mapper --database_path $PROJECT/database.db --image_path $PROJECT/images --output_path $PROJECT/sparse

colmap model_converter --input_path $PROJECT/sparse/0 --output_path $PROJECT/sparse --output_type TXT

InterfaceCOLMAP --working-folder $PROJECT --input-file $PROJECT --output-file $PROJECT/model_colmap.mvs --archive-type 0 

DensifyPointCloud --input-file $PROJECT/model_colmap.mvs --working-folder $PROJECT --output-file $PROJECT/model_dense.mvs --archive-type 0 -v 0

ReconstructMesh --input-file $PROJECT/model_dense.mvs --working-folder $PROJECT/ --output-file $PROJECT/model_dense_mesh.mvs --archive-type 0

RefineMesh --resolution-level 1 --input-file $PROJECT/model_dense_mesh.mvs --working-folder $PROJECT/ --output-file $PROJECT/model_dense_mesh_refine.mvs --archive-type 0

TextureMesh --export-type obj --output-file $PROJECT/model.obj --working-folder $PROJECT/ --input-file $PROJECT/model_dense_mesh_refine.mvs --archive-type 0 -v 0