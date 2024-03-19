#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=log_gradient3
#SBATCH --output=log_gradient3.out
#SBATCH --error=log_gradient3.err

#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=24G

job_name="log_gradient3"

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

# rm single_image.out
# rm single_image.err
rm -rf checkpoints_experiment/$job_name
rm -rf saved_models_experiment/$job_name
echo "removed saved folders"

python train.py \
--num-scales 1 \
-b1 -s0.1 -c0.5 --sequence-length 3 \
--learning-rate 1e-4 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name $job_name \
--depth_model dispnet \
--train depth \
--use_gt_pose \
--use_gt_mask \
--epoch-size 10 \




# --pretrained-pose /mundus/mrahman527/Thesis/saved_models/do_not_optimize_s0.5_c1_sl3/exp_pose_model_best.pth.tar \
