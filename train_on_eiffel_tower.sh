#!/bin/bash

#SBATCH --time=2-00:01:00
#SBATCH --job-name=train_Eiffel
#SBATCH --output=train_Eiffel.out
#SBATCH --error=train_Eiffel.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=24G


echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train.py /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--resnet-layers 50 \
--num-scales 1 \
-b32 -s0.1 -c0.5 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--name downscaled_eiffel_tower_resnet50_batch32