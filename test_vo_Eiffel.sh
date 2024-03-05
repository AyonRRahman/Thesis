#!/bin/bash

#SBATCH --time=02:00:00
#SBATCH --job-name=test_vo
#SBATCH --output=test_vo.out
#SBATCH --error=test_vo.err
#SBATCH --gres=gpu

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

echo "saved_models/compare_new_train_with_mask"
python test_vo.py --pretrained_dir saved_models/compare_new_train_with_mask \
--dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--name saved_models/compare_new_train_with_mask \
--use_best

echo "compare_new_train_without_mask"
python test_vo.py --pretrained_dir saved_models/compare_new_train_without_mask \
--dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--name compare_new_train_without_mask \
--use_best

echo "saved_models/do_not_optimize_s0.5_c1_sl3"
python test_vo.py --pretrained_dir saved_models/do_not_optimize_s0.5_c1_sl3 \
--dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
--name saved_models/do_not_optimize_s0.5_c1_sl3 \
--use_best