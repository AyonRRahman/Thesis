#!/bin/bash

#SBATCH --time=04:00:00
#SBATCH --job-name=eval_depth
#SBATCH --output=eval_depth.out
#SBATCH --error=eval_depth.err
#SBATCH --partition=mundus
#SBATCH --gres=gpu


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

# echo "disp_net_with_mask_comparison"
# python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/compare_new_train_with_mask/dispnet_model_best.pth.tar

# echo "disp_net_without_mask_comparison"
# python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar


# echo "equal weight"
# python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/equal_wrights_b16_sl3_lr1e-4/dispnet_model_best.pth.tar

#udepth 
echo "===================="

echo "RGB"
# echo "checkpoint"
# python eval_depth.py --scale --depth_model udepth --saved_model saved_models/udepth_rmi_train/dispnet_checkpoint.pth.tar
echo "===================="

echo "best"
# python eval_depth.py --scale --depth_model udepth --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar --max_depth 25
echo "===================="


echo "RMI"

# echo "checkpoint"
# python eval_depth.py --scale --depth_model udepth --saved_model saved_models/udepth_actual_rmi_train/dispnet_checkpoint.pth.tar --RMI --max_depth 25

start_value=20
end_value=100
increment=5

# Loop through each integer value from start to end with the specified increment
for ((i=start_value; i<=end_value; i+=increment)); do
    echo "best: $i" 
    python eval_depth.py --scale --depth_model udepth --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar --RMI --max_depth $i
    python eval_depth.py --scale --depth_model udepth --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar --max_depth $i
    python eval_depth.py --scale --depth_model dispnet --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar --max_depth $i
done

# echo "================"
# echo "Varos"
# echo "========================"

# echo "RMI"

# echo "checkpoint"
# python eval_depth.py --scale --depth_model udepth \
# --saved_model saved_models/udepth_actual_rmi_train/dispnet_checkpoint.pth.tar --RMI \
# --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
# --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

# echo "======================="
# echo "best"
# python eval_depth.py --scale --depth_model udepth \
# --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar --RMI \
# --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
# --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img


# echo "===================="

# echo "RGB"
# echo "checkpoint"
# python eval_depth.py --scale --depth_model udepth \
# --saved_model saved_models/udepth_rmi_train/dispnet_checkpoint.pth.tar \
# --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
# --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

# echo "===================="

# echo "best"
# python eval_depth.py --scale --depth_model udepth \
# --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
# --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
# --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

# echo "===================="

