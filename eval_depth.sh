#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --job-name=eval_depth
#SBATCH --output=eval_depth.out
#SBATCH --error=eval_depth.err

#SBATCH --gres=gpu
#SBATCG --mem-per-gpu=24G


echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

# echo "dpts"
# python eval_depth.py --scale --depth_model dpts

# echo "dptl"
# python eval_depth.py --scale --depth_model dptl

# echo "dptb"
# python eval_depth.py --scale --depth_model dptb

echo "do_not_optimize_do_not_optimize_s0.5_c1_sl3"
python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/do_not_optimize_s0.5_c1_sl3/dispnet_model_best.pth.tar
