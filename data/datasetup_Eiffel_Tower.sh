#!/bin/bash

echo 'Downloading Eiffel Tower Dataset'
# Download all year's data for the Eiffel Tower dataset
python3 download_Eiffel_Tower_data.py download --dataset Eiffel-Tower --delete_archive

echo 'Using colmap to undistort Dataset'
#undistort the Eiffel tower dataset using colmap to use pinhole model

mkdir Eiffel-Tower_undistorted
python3 undistort_Eiffel_Tower_data.py 