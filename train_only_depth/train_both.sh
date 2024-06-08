#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=calc_err2
#SBATCH --output=calc_err2.out
#SBATCH --error=calc_err2.err

#SBATCH --gres=gpu:a40-48:1



echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME


which python

python train.py --data /home/mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled_colmap \
--num-scales 1 \
-b32 -s0.1 -c1 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 --with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name test_pose_inconsistency3 \
--depth_model dispnet \
--train both --train_until_converge \
--epoch-size 0 \
--pretrained-disp /home/mundus/mrahman527/Thesis/saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar \
--pretrained-pose /home/mundus/mrahman527/Thesis/saved_models/compare_new_train_without_mask/exp_pose_model_best.pth.tar


# --pretrained-pose /mundus/mrahman527/Thesis/saved_models/do_not_optimize_s0.5_c1_sl3/exp_pose_model_best.pth.tar \
