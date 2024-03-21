#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=udepth_pretrained2
#SBATCH --output=udepth_pretrained2.out
#SBATCH --error=udepth_pretrained2.err
#SBATCH --gres=gpu:a40-48:1
#SBATCH --mem-per-gpu=48G

which python

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train_new.py data/Eiffel-Tower_ready_Downscaled_colmap \
./data/scaled_and_cropped_mask/ \
--num-scales 1 \
-b32 -s0.1 -c0.5 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name udepth_pretrained2 \
--depth_model dpts \
# --epoch-size 20
# --pretrained-pose /mundus/mrahman527/Thesis/saved_models/do_not_optimize_s0.5_c1_sl3/exp_pose_model_best.pth.tar \
