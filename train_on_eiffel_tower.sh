#!/bin/bash

#SBATCH --time=7-00:00:00
#SBATCH --job-name=train_untill_converge
#SBATCH --output=train_untill_converge.out
#SBATCH --error=train_untill_converge.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=raqibur.ayon@gmail.com
#SBATCH --gres=gpu
#SBATCH --mem-per-gpu=24G
#SBATCH --partition=besteffort


echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

python train_new.py /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--num-scales 1 \
-b16 -s0.5 -c1 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output \
--use_pretrained \
--epochs 400 --learning-rate 1e-4 \
--name train_untill_converge \
