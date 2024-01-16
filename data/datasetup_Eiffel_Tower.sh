#!/bin/bash

# Download all year's data for the Eiffel Tower dataset
python3 download_Eiffel_Tower_data.py download --dataset Eiffel-Tower --delete_archive

#undistort the Eiffel tower dataset using colmap to use pinhole model
mkdir Eiffel-Tower_undistorted
python3 undistort_Eiffel_Tower_data.py 