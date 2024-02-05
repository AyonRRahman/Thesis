#!/bin/bash

#uses maxime's script to generate depth images
# Path to the data folder
data_folder="./Eiffel-Tower_undistorted"

# Define the subfolders
folders=("2015" "2016" "2018" "2020")


# Loop through each folder and list its contents
for folder in "${folders[@]}"; do
    echo "$folder"
    chmod 777 "$data_folder/$folder/scene.ply"
    mkdir $data_folder/$folder/depth_images

    ../Thirdparty/depth_map_2_mesh_ray_tracer/build/ply_mesh_to_depth_maps $data_folder/$folder/scene.ply $data_folder/$folder/sparse $data_folder/$folder/depth_images
    echo ""
done