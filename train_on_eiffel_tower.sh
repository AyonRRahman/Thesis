#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=depth_any_without_mask
#SBATCH --output=depth_any_without_mask.out
#SBATCH --error=depth_any_without_mask.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com
#SBATCH --gres=gpu:a100-80:1
#SBATCH --mem-per-gpu=80G

which python

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train_new.py data/Eiffel-Tower_ready_Downscaled_colmap \
./data/scaled_and_cropped_mask/ \
--num-scales 1 \
-b8 -s0.1 -c0.5 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--pretrained-pose /mundus/mrahman527/Thesis/saved_models/do_not_optimize_s0.5_c1_sl3/exp_pose_model_best.pth.tar \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name depth_any_without_mask \
--depth_model dpts
