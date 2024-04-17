import sys
import os
import argparse
from path import Path
import shutil
import numpy as np
from tqdm import tqdm

from utils.data import get_cam_txt

parser = argparse.ArgumentParser(description='Prepare Varos Dataset for SC-SFMLearner')

parser.add_argument('--dir', type=str, default='/home/mundus/mrahman527/Varos',
                    help='Directory path of the datasets' 
                    )
parser.add_argument('--out', type=str, default='data/Varos_ready', help='Directory to save the undistorted dataset')

def prepare_Varos_dataset():
    '''
    Prepares Varos Dataset as needed for SC-SFMLearner.
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
    
    images = (Path(args.dir)/'2021-08-17_SEQ1'/'vehicle0'/'cam0'/'A').glob('*.png')
    output_dir = Path(args.out)
    folder = output_dir/'seq01'
    folder.makedirs_p()
    
    for image in tqdm(images):
        shutil.copy2(image, folder)
            
    assert len(os.listdir(folder))==len(images)

    #get the camera intrinsic
    #save the cam.txt file 
    
    camera_intrinsics = np.array([[985.5072, 0, 640],
                                  [0, 985.5072, 360],
                                  [0, 0, 1]]) #from paper

    np.savetxt(os.path.join(folder, 'cam.txt'), camera_intrinsics)
            
    with open(os.path.join(args.out, 'val.txt'), 'w') as file:
        file.write("seq01\n")
    
    print('Done creating val.txt files')
    
    
if __name__=='__main__':
    # prepare_Varos_dataset()
    print(len(os.listdir('data/Varos_ready/seq01')))

    print(len(os.listdir('/home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A')))