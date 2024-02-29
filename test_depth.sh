#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --job-name=test_depth
#SBATCH --output=test_depth.out
#SBATCH --error=test_depth.err

#SBATCH --gres=gpu



echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

# python test_vo.py --pretrained_dir saved_models/do_not_optimize_s0.5_c1_sl3 \
# --dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
# --name do_not_optimize_s0.5_c1_sl3 \
# --use_best

# python test_vo.py --pretrained_dir saved_models/do_not_optimize_s0.5_c1_sl3 \
# --dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
# --name do_not_optimize_s0.5_c1_sl3 \

# python test_vo.py --pretrained_dir saved_models/equal_wrights_b16_sl3_lr1e-4 \
# --dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
# --name equal_wrights_b16_sl3_lr1e-4 \
# --use_best

# python test_vo.py --pretrained_dir saved_models/equal_wrights_b16_sl3_lr1e-4 \
# --dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
# --name equal_wrights_b16_sl3_lr1e-4 \

python test_depth.py --pretrained_dir saved_models/equal_wrights_b16_sl3_lr1e-4 \
--dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--name equal_wrights_b16_sl3_lr1e-4 \

