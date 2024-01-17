import sys
import os 

#setting the path to the project root
script_path = os.path.abspath(__file__)
project_folder = os.path.abspath(os.path.join(os.path.dirname(script_path), '..'))
sys.path[0] = project_folder

# print(sys.path)
import numpy as np
from utils.colmap_script import read_cameras_binary

def get_cam_txt(dir: str) -> np.ndarray:
    '''
    This function converts the cameras.bin file generated by colmap undistort image command
    and creates the intrinsic camera matrix and saves it as cam.txt file.

    params:
        dir: path to the folder where colmap saved after undistorting images.
            should contain images, sparse etc. folder
    
    Returns:
        intrinsic_matrix: 3x3 np array representing camera intrinsic matrix
    
    '''
    try:
        assert  os.path.exists(os.path.join(dir, 'sparse/cameras.bin'))
    except:
        print(f"{os.path.join(dir, 'sparse/cameras.bin')} does not exist.")
        raise ValueError(f"Folder {os.path.join(dir, 'sparse/cameras.bin')} does not exist.")


    cameras = read_cameras_binary(os.path.join(dir, 'sparse/cameras.bin'))
    for x in cameras:
        assert cameras[x].model=='PINHOLE'
        assert len(cameras[x].params)==4

        fx = cameras[x].params[0]
        fy = cameras[x].params[1]
        cx = cameras[x].params[2]
        cy = cameras[x].params[3]

        intrinsic_matrix = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
        return intrinsic_matrix
        # np.savetxt(os.path.join(dir, 'images/cam.txt'), intrinsic_matrix)

        


if __name__=="__main__":
    print('!!!!!!!')
    # Get the absolute path of the current script
    # script_path = os.path.abspath(__file__)
    # # Construct the path to the project folder based on the script's location
    # Project_folder = os.path.abspath(os.path.join(os.path.dirname(script_path), '..', '..'))
    
    # # Add the 'Thesis' folder to sys.path
    # # sys.path.append(thesis_folder)
    # print(sys.path)