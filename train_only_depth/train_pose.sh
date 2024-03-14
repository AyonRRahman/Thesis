#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=pose_train
#SBATCH --output=pose_train.out
#SBATCH --error=pose_train.err

#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=24G

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME



python train.py \
--num-scales 1 \
-b8 -s0.1 -c0.5 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 --with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name pose_train \
--depth_model dispnet \
--train pose \
--use_gt_depth \
--use_gt_mask \



# --pretrained-pose /mundus/mrahman527/Thesis/saved_models/do_not_optimize_s0.5_c1_sl3/exp_pose_model_best.pth.tar \
