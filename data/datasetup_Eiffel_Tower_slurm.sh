#!/bin/bash

#SBATCH --time=2-00:01:00
#SBATCH --job-name=datasetup_Eiffel
#SBATCH --output=datasetup_Eiffel.out
#SBATCH --error=datasetup_Eiffel.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com

module load colmap
module load cuda/11.0 

echo '---------------'
which python
conda init
conda activate thesis
which python
echo '---------------'

echo $HOSTNAME
#Eiffel Tower
# ./datasetup_Eiffel_Tower.sh

echo $HOSTNAME

# Kitty
cd data
mkdir Kitty_raw
cp Kitty_raw_data_downloader.sh Kitty_raw
cd Kitty_raw
./Kitty_raw_data_downloader.sh

echo $HOSTNAME