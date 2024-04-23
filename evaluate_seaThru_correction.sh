#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=eval_seaThru
#SBATCH --output=eval_seaThru.out
#SBATCH --error=eval_seaThru.err
#SBATCH --partition=cpu

which python

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

echo "udepth_rmi_eiffel"
python evaluate_seaThru_correction.py --depth_model udepth --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar --RMI --save_freq 10 --name udepth_rmi_eiffel

echo "udepth_rgb_eiffel"
python evaluate_seaThru_correction.py --depth_model udepth --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar --save_freq 10 --name udepth_rgb_eiffel

echo "dispnet_rgb_eiffel"
python evaluate_seaThru_correction.py --depth_model dispnet --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  --save_freq 10 --name dispnet_rgb_eiffel

#Varos
echo "udepth_rmi_varos"
python evaluate_seaThru_correction.py --depth_model udepth --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar --RMI --save_freq 10 --name udepth_rmi_varos \
--gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
--img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

echo "udepth_rgb_varos"
python evaluate_seaThru_correction.py --depth_model udepth --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar --save_freq 10 --name udepth_rgb_varos \
--gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
--img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

echo "dispnet_rgb_varos"
python evaluate_seaThru_correction.py --depth_model dispnet --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  --save_freq 10 --name dispnet_rgb_varos \
--gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
--img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

