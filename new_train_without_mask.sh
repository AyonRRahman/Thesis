#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=compare_new_train_without_mask
#SBATCH --output=compare_new_train_without_mask.out
#SBATCH --error=compare_new_train_without_mask.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=16G



echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train_new.py /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled_colmap/ \
./data/scaled_and_cropped_mask \
--num-scales 1 \
-b16 -s0.1 -c1 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name compare_new_train_without_mask \
--train_until_converge \

