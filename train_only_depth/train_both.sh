#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=attn_mask
#SBATCH --output=attn_mask.out
#SBATCH --error=attn_mask.err

#SBATCH --gres=gpu:a40-48:1

#SBATCH --partition=gpu

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME


which python

python train.py \
--num-scales 1 \
-b32 -s0.1 -c0.5 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 --with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name attn_mask \
--depth_model udepth \
--train both \
--use_gt_depth \
--use_gt_mask \
--use_gt_pose --use_RMI --use_mask_from_attn 



# --pretrained-pose /mundus/mrahman527/Thesis/saved_models/do_not_optimize_s0.5_c1_sl3/exp_pose_model_best.pth.tar \
