import sys
import os
import argparse

import shutil
import numpy as np

from utils.data import get_cam_txt

parser = argparse.ArgumentParser(description='Prepare Eiffel Tower Dataset for SC-SFMLearner')

parser.add_argument('--dir', type=str, default='data/Eiffel-Tower_undistorted',
                    help='Directory path of the datasets (default: Eiffel-Tower_undistorted)' 
                    )
parser.add_argument('--out', type=str, default='data/Eiffel-Tower_ready', help='Directory to save the undistorted dataset')
parser.add_argument('--delete_archive', action='store_true', help='Delete the archive after processing')

def prepare_Eiffel_tower_dataset():
    '''
    Prepares Eiffel Tower Dataset as needed for SC-SFMLearner.
    '''
    args = parser.parse_args()
    
    try:
        assert os.path.exists(args.dir)
    except AssertionError as e:
        print(f"{args.dir} Directory not found")
        return

    try:    
        assert os.path.exists(args.out)

    except AssertionError as e:
        print(f"{args.out} directory not found. Making new folder {args.out}")
        os.mkdir(args.out)
    
    years = os.listdir(args.dir)
    
    for year in years:
        target_folder = os.path.join(args.out, year)
        
        src_folder = os.path.join(args.dir, year, 'images')
        print(f"copying {year} folder")
        
        #check the files already exists or not
        if len(os.listdir(src_folder)) + 1 == len(os.listdir(target_folder)):
            print("All files present. Skipping")
            continue

        #copy the images
        shutil.copytree(src_folder, target_folder)
        
        assert len(os.listdir(src_folder))==len(os.listdir(target_folder))

        #get the camera intrinsic
        camera_intrinsics = get_cam_txt(os.path.join(args.dir, year)) #camera intrinsic matrix
        #save the cam.txt file 
        np.savetxt(os.path.join(target_folder, 'cam.txt'), camera_intrinsics)
        

    #making the val.txt and train.txt files
    with open(os.path.join(args.out, 'train.txt'), 'w') as file:
        file.write("2018\n")
        file.write("2016\n")
        file.write("2020")

    with open(os.path.join(args.out, 'val.txt'), 'w') as file:
        file.write("2015")
    print('Done creating train.txt and val.txt files')

if __name__=='__main__':
    prepare_Eiffel_tower_dataset()

