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
cd ..
python prepare_Eiffel_tower.py
```
Or undistort using opencv to keep the image dimension same and prepare for training:
```bash
cd data
python undistort_Eiffel_tower_using_opencv.py
```
## Creating Depth Maps from SFM files for validation
Using these repositories:
[Maxime Ferrera's Script](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/tree/main) and [OpenMVS](https://github.com/cdcseacave/openMVS/tree/master).<br>
Build [Maxime Ferrera's Script](https://github.com/ferreram/depth_map_2_mesh_ray_tracer/tree/main) according to the mentioned method in the repository in the thirdparty folder. <br>
Use docker file given in the [OpenMVS](https://github.com/cdcseacave/openMVS/tree/master) to build a docker container.
```bash
cd Thirdparty/openMVS/docker/

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

