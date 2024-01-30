import os
import argparse
from path import Path
import numpy as np
import cv2
from tqdm import tqdm

def get_camera_params(camera_file: Path) -> (np.ndarray, np.ndarray):
    '''
    opens the camera.txt file and returns the camera 
    calibration matrix and distortion coefficients.
    Params:
        camera_file (Path): path to the file camera.txt
    returns:
        camera_calibration_mtx (np.ndarray): 3x3 matrix for the camera parameter
        dist (np.ndarray): distortion coefficients
    '''    

    #opening the camera params file
    with open(camera_file) as f:
        for x in f:
            if x[0]=='#':
                continue
            cam_params = x.split(' ')[4:]
            fl_x = float(cam_params[0])
            fl_y = float(cam_params[0])
            cx = float(cam_params[1])
            cy = float(cam_params[2])
            k1 = float(cam_params[3])
            k2 = float(cam_params[4])
            p1 = 0.0
            p2 = 0.0
    
    camera_calibration_mtx = np.array([[fl_x, 0, cx],[0, fl_y, cy],[0, 0, 1]])
    dist = np.array([k1, k2, p1, p2]) 

    return camera_calibration_mtx, dist

def main():
    '''
    Undistort Eiffel Tower data using opencv and prepare for training.
    '''

    parser = argparse.ArgumentParser(description='Undistort Eiffel Tower images using Opencv.')

    parser.add_argument('--dir', type=str, default='Eiffel-Tower', help='Directory path of the datasets (default: Eiffel_tower)')
    parser.add_argument('--out', type=str, default='Eiffel-Tower_ready_opencv', help='Directory to save the undistorted dataset')
    
    # Parse the command-line arguments
    args = parser.parse_args()

    root = Path(args.dir)

    try:
        assert os.path.isdir(root)
    except:
        raise ValueError(f"Folder {root} does not exist.")
    
    #make the output folder if it doesnot exist
    output_path = Path(args.out)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    years = os.listdir(root)
    for year in years:
        print(f'processing {year} folder')
        #sorting the files and folders
        year_folder = root/year
        output_year_folder = output_path/year

        if not os.path.exists(output_year_folder):
            os.mkdir(output_year_folder)
    
        camera_file = year_folder/'sfm/cameras.txt'
        images_folder = year_folder/'images'
        img_list = os.listdir(images_folder)

        camera_calibration_mtx, dist = get_camera_params(camera_file)
        #saving the cam.txt file
        
        for image in tqdm(sorted(img_list)):
            img = cv2.imread(images_folder/image)
            undistorted_img = cv2.undistort(img, camera_calibration_mtx, dist, None, camera_calibration_mtx)
            cv2.imwrite(output_year_folder/image, undistorted_img)
            
            
        np.savetxt(output_year_folder/'cam.txt', camera_calibration_mtx)
        print(f'Done processing {year}')
    
    #saving the train.txt and val.txt file
    with open(output_path/'train.txt', 'w') as file:
        file.write("2018\n")
        file.write("2016\n")
        file.write("2020\n")

    with open(output_path/'val.txt', 'w') as file:
        file.write("2015\n")
    print('Done saving training.txt and val.txt files')
        
if __name__=='__main__':
    main()