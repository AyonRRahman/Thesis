#!/bin/bash

#SBATCH --time=4-00:01:00
#SBATCH --job-name=train_Eiffel_new_weight
#SBATCH --output=train_Eiffel_new_weight.out
#SBATCH --error=train_Eiffel_new_weight.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=40G


echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train.py /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--resnet-layers 50 \
--num-scales 1 \
-b16 -s0.5 -c1 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name downscaled_eiffel_tower_resnet50_batch16_with_s0.5_c1