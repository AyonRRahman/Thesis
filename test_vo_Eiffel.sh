#!/bin/bash

#SBATCH --time=18:00:00
#SBATCH --job-name=test_vo
#SBATCH --output=test_vo.out
#SBATCH --error=test_vo.err
#SBATCH --gres=gpu

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

# --dataset-dir data/Varos_ready  --name udepth_rmi_varos --use_best --dataset varos --sequence seq01  


echo "udepth_rmi"
python test_vo.py --pretrained_dir saved_models/udepth_actual_rmi_train/ \
--dataset-dir data/Varos_ready --dataset varos --sequence seq01 \
--name udepth_rmi_varos \
--use_best


echo "udepth_rgb"
python test_vo.py --pretrained_dir saved_models/udepth_rmi_train/ \
--dataset-dir data/Varos_ready --dataset varos --sequence seq01 \
--name udepth_rgb_varos \
--use_best

echo "dispnet_mvit_rmi"
python test_vo.py --pretrained_dir saved_models/dispnet_mvit_rmi/ \
--dataset-dir data/Varos_ready --dataset varos --sequence seq01 \
--name dispnet_mvit_rmi_varos \
--use_best

echo "dispnet_mvit_rgb"
python test_vo.py --pretrained_dir saved_models/dispnet_mvit_rgb/ \
--dataset-dir data/Varos_ready --dataset varos --sequence seq01 \
--name dispnet_mvit_rgb_varos \
--use_best

echo "dispnet_rmi"
python test_vo.py --pretrained_dir saved_models/dispnet_rmi/ \
--dataset-dir data/Varos_ready --dataset varos --sequence seq01 \
--name dispnet_rmi_varos \
--use_best


echo "dispnet_rgb"
python test_vo.py --pretrained_dir saved_models/compare_new_train_without_mask/ \
--dataset-dir data/Varos_ready --dataset varos --sequence seq01 \
--name dispnet_rgb_varos \
--use_best



# echo "compare_new_train_without_mask"
# python test_vo.py --pretrained_dir saved_models/compare_new_train_without_mask \
# --dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
# --name compare_new_train_without_mask \
# --use_best

# echo "saved_models/do_not_optimize_s0.5_c1_sl3"
# python test_vo.py --pretrained_dir saved_models/do_not_optimize_s0.5_c1_sl3 \
# --dataset-dir /mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled/ \
# --name saved_models/do_not_optimize_s0.5_c1_sl3 \
# --use_best

