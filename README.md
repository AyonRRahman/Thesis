## Installation
Clone the repository: 
```bash
git clone --recurse-submodules git@github.com:AyonRRahman/Thesis.git
```
## Data
Download the Eiffel Tower dataset and use colmap to undistort images
```bash
cd data
chmod +x datasetup_Eiffel_Tower.sh
./datasetup_Eiffel_Tower.sh
```
Prepare the dataset for training in SC-SFMlearner:
```bash
run the crop_and_downscale_colmap_undistorted_image() function in utils.data.py to prepare the the Eiffel-Tower dataset for training.
```
Create the ground truth trajectory using utils.utils.export_trajectory() function in the kitty format.

Or undistort using opencv to keep the image dimension same and prepare for training:
```bash
cd data
python undistort_Eiffel_tower_using_opencv.py
```
use Downscale_image function from utils.data to downscale the images.
## Creating Depth Maps from SFM files for validation
Using these repositories:
[Maxime Ferrera's Script](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/tree/main) and [OpenMVS](https://github.com/cdcseacave/openMVS/tree/master).<br>
Build [Maxime Ferrera's Script](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/tree/main) according to the mentioned method in the repository in the thirdparty folder. <br>
Use docker file given in the [OpenMVS](https://github.com/cdcseacave/openMVS/tree/master) to build a docker container.

convert the colmap model and start the docker for openMVS in the undistorted(colmap) folder
```bash
python create_txt_files_for_colmap_model.py
../Thirdparty/openMVS/docker/QUICK_START.sh ./Eiffel-Tower_undistorted
```
In the terminal of the docker container for each folder(2020, 2015, 2016, 2018) of data do this:
```bash
InterfaceCOLMAP -i ./2020 -o ./2020/scene.mvs --image-folder ./2020/images
ReconstructMesh ./2020/scene.mvs -o ./2020/scene.ply
```
now from the native terminal:
```bash
sudo ./Create_depth_images.sh 
```
## Download the Kitty Raw dataset
Official link to download the dataset:
http://www.cvlibs.net/download.php?file=raw_data_downloader.zip
```bash
cd data
mkdir Kitty_raw
cp Kitty_raw_data_downloader.sh Kitty_raw
cd Kitty_raw
./Kitty_raw_data_downloader.sh

```

