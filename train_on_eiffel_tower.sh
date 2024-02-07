#!/bin/bash

#SBATCH --time=4-00:01:00
#SBATCH --job-name=sigmoid_new_training_optimize_weight_b16_sl3_lr1e-4
#SBATCH --output=sigmoid_new_training_optimize_weight_b16_sl3_lr1e-4.out
#SBATCH --error=sigmoid_new_training_optimize_weight_b16_sl3_lr1e-4.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=24G


echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train_new.py /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--num-scales 1 \
-b16 -s1 -c1 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--optimize_loss_weight \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name sigmoid_new_training_optimize_weight_b16_sl3_lr1e-4