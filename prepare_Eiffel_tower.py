import os
import argparse

from utils.data import get_cam_txt

parser = argparse.ArgumentParser(description='Prepare Eiffel Tower Dataset for SC-SFMLearner')

parser.add_argument('--dir', type=str, default='Eiffel-Tower_undistorted', help='Directory path of the datasets (default: Eiffel-Tower_undistorted)')
parser.add_argument('--out', type=str, default='Eiffel-Tower', help='Directory to save the undistorted dataset')
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
    
    

if __name__=='__main__':
    prepare_Eiffel_tower_dataset()

