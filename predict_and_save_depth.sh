#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=predict_and_save_depth
#SBATCH --output=predict_and_save_depth.out
#SBATCH --error=predict_and_save_depth.err
#SBATCH --partition=cpu

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

which python

python predict_and_save_depth.py --depth_model udepth \
--saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
--RMI --name udepth_rmi_eiffel