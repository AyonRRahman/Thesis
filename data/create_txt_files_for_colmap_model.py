# first use model model_converter to convert the bin files to txt files
# colmap model_converter \
#     --input_path path-to-binary-reconstruction \
#     --output_path path-to-txt-reconstruction \
#     --output_type TXT

# then use openmvs to create the mesh with .ply extension

#then use the maxime's script to create the depth images.

import os
from path import Path
import sys 
import subprocess
import argparse 

parser = argparse.ArgumentParser(description='Run colmap image_undistorter on Eiffel Tower dataset.')

parser.add_argument('--dir', type=str, default='Eiffel-Tower_undistorted', help='Directory path of the datasets (default: Eiffel_tower)')
# parser.add_argument('--out', type=str, default='Eiffel-Tower_undistorted', help='Directory to save the undistorted dataset')
# parser.add_argument('--delete_archive', action='store_true', help='Delete the archive after processing')
colmap_command_template = "colmap model_converter --input_path {}/sparse_bin \
        --output_path {}/sparse \
        --output_type TXT"
mv_command_template = "mv  {}/sparse {}/sparse_bin"
    
def colmap_converter(args, folder):
    
    root = Path(args.dir) 

    folder_path = root/folder
    mv_command = mv_command_template.format(folder_path, folder_path)
    # print(mv_command)
    # print(os.path.isdir(Path(args.dir+'/'+folder+'/sparse')))
    
    try:
        subprocess.run(mv_command, shell=True, check=True)
        print(f"move Command executed successfully for dataset {folder}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing move command for dataset {folder}: {e}")
    
    colmap_command = colmap_command_template.format(folder_path, folder_path)
    
    (folder_path/'sparse').makedirs_p()
    try:
        subprocess.run(colmap_command, shell=True, check=True)
        print(f"Colmap Command executed successfully for dataset {folder}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing Colmap command for dataset {folder}: {e}")
    

# def openmvs_mesh_create(args):
#     script_path = os.path.abspath(__file__)
#     project_folder = os.path.abspath(os.path.join(os.path.dirname(script_path), '..'))
#     project_folder = Path(project_folder)
#     openmvs_folder = project_folder/'Thirdparty'/'openMVS'
#     docker_folder = openmvs_folder/'docker'
#     # print(os.listdir(docker_folder))
#     # docker_folder/'QUICK_START.sh /path/where/your/SFM/results/are'
#     docker_file = docker_folder/'QUICK_START.sh'
#     docker_starting_folder = Path(os.path.dirname(script_path))/args.dir
#     docker_start_command = docker_file + ' '+ docker_starting_folder
#     print(docker_start_command)

#     docker_process = subprocess.Popen(["bash", "-c", docker_start_command], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


#     # Send the command to the Docker terminal
#     docker_process.stdin.write("ls -l\n".encode())
#     docker_process.stdin.close()

#     # Read the output of the command
#     output, errors = docker_process.communicate()
    
#     # Print the output and errors
#     print("Output:", output.decode())
#     print("Errors:", errors.decode())

#     pass

def main():
    args = parser.parse_args()
    data_folders = os.listdir(args.dir)
    # print(data_folders)
    
    for folder in data_folders:
        colmap_converter(args, folder)
    
    # openmvs_mesh_create(args)

if __name__=='__main__':
    main()



