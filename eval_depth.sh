#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --job-name=eval_depth
#SBATCH --output=eval_depth.out
#SBATCH --error=eval_depth.err

#SBATCH --gres=gpu
#SBATCG --mem-per-gpu=24G

which python

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

# echo "dpts"
# python eval_depth.py --scale --depth_model dpts

# echo "dptl"
# python eval_depth.py --scale --depth_model dptl

# echo "dptb"
# python eval_depth.py --scale --depth_model dptb

echo "disp_net_with_mask_comparison"
python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/compare_new_train_with_mask/dispnet_model_best.pth.tar

echo "disp_net_without_mask_comparison"
python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar
