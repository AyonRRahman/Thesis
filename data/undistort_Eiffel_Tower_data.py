import subprocess
import os
import argparse

def main():
    '''
    Undistort Eiffel Tower data using colmap and make a pinhole camera models
    '''
    parser = argparse.ArgumentParser(description='Run colmap image_undistorter on Eiffel Tower dataset.')

    parser.add_argument('--dir', type=str, default='Eiffel-Tower', help='Directory path of the datasets (default: Eiffel_tower)')
    parser.add_argument('--out', type=str, default='Eiffel-Tower_undistorted', help='Directory to save the undistorted dataset')
    parser.add_argument('--delete_archive', action='store_true', help='Delete the archive after processing')

    # Parse the command-line arguments
    args = parser.parse_args()

    
    datasets = os.listdir(args.dir)

    command_template = "colmap image_undistorter --image_path '{}/{}/images' --input_path '{}/{}/sfm' --output_path '{}/{}' --output_type COLMAP"

    # Iterate through each dataset
    for dataset in datasets:
        # Create the full command by formatting the template
        command = command_template.format(args.dir, dataset, args.dir, dataset, args.out, dataset)
        
        
        try:
            subprocess.run(command, shell=True, check=True)
            # print(command)
            print(f"Command executed successfully for dataset {dataset}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error executing command for dataset {dataset}: {e}")
            return
        

    #delete the dataset to save space
    if args.delete_archive and len(os.listdir(args.out))==len(datasets):
        delete_command = f"rm -rf {args.dir}"
        try:
            subprocess.run(delete_command, shell=True, check=True)
            # print(delete_command)
            print(f'Successfully deleted {args.dir}')
        except subprocess.CalledProcessError as e:
            print(f"Error executing command {e}")

if __name__=='__main__':
    main()