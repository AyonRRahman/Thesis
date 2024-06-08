#!/bin/bash

#SBATCH --time=7-00:00:00 --partition=all
#SBATCH --job-name=eval_seaThru_varos
#SBATCH --output=eval_seaThru_varos.out
#SBATCH --error=eval_seaThru_varos.err


which python

echo $HOSTNAME
nvidia-smi 
echo $HOSTNAME

# python evaluate_seaThru_correction.py --depth_model udepth --save_freq 1 --name eiffel_gt_depth --use_gt_depth

python evaluate_seaThru_correction.py --depth_model udepth --save_freq 1 --name varos_gt_depth --use_gt_depth \
--img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A \
--gt_depth /home/mundus/mrahman527/Thesis/data/Varos_ready/depth/seq01 --rescale_img


# Define the commands in a loop
# for i in {1..8}; do   # Adjust the number of iterations to match the array size
    # echo "Job $SLURM_ARRAY_TASK_ID"
    # case $SLURM_ARRAY_TASK_ID in
    #     1)
    #         echo "udepth_rmi_eiffel"
    #         python evaluate_seaThru_correction.py --depth_model udepth \
    #         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
    #         --RMI --save_freq 10 --name udepth_rmi_eiffel
    #         ;;
    #     2)
    #         echo "udepth_rgb_eiffel"
    #         python evaluate_seaThru_correction.py --depth_model udepth \
    #         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
    #         --save_freq 10 --name udepth_rgb_eiffel
    #         ;;
    #     3)
    #         echo "dispnet_rgb_eiffel"
    #         python evaluate_seaThru_correction.py --depth_model dispnet \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_rgb_eiffel
    #         ;;
    #     4)
    #         echo "dispnet mvit rgb eiffel"
    #         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_mvit_rgb_eiffel
    #         ;;
    #     5)
    #         echo "udepth_rmi_varos"
    #         python evaluate_seaThru_correction.py --depth_model udepth \
    #         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
    #         --RMI --save_freq 10 --name udepth_rmi_varos --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
    #         ;;
    #     6)
    #         echo "udepth_rgb_varos"
    #         python evaluate_seaThru_correction.py --depth_model udepth \
    #         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
    #         --save_freq 10 --name udepth_rgb_varos \
    #         --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
    #         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
    #         ;;
    #     7)
    #         echo "dispnet_rgb_varos"
    #         python evaluate_seaThru_correction.py --depth_model dispnet \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_rgb_varos \
    #         --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
    #         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
    #         ;;
    #     8)
    #         echo "dispnet mvit rgb varos"
    #         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_mvit_rgb_varos \
    #         --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
    #         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
    #         ;;
        
    #     9)
    #         echo "dispnet_rmi_eiffel"
    #         python evaluate_seaThru_correction.py --depth_model dispnet \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_rmi_eiffel --RMI
    #         ;;
    #     10)
    #         echo "dispnet mvit rmi eiffel"
    #         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_mvit_rmi_eiffel --RMI
    #         ;;
    #     11)
    #         echo "dispnet_rmi_varos"
    #         python evaluate_seaThru_correction.py --depth_model dispnet \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_rmi_varos --RMI \
    #         --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
    #         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
            
            
    #         ;;
    #     12)
    #         echo "dispnet mvit rmi varos"
    #         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
    #         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
    #         --save_freq 10 --name dispnet_mvit_rmi_varos --RMI\
    #         --gt_depth /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/D \
    #         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
            
    #         ;;
    # esac


# done


# folders=("Curasao" "IUI3-RedSea" "JapaneseGradens-RedSea" "Panama")


# echo "Job $SLURM_ARRAY_TASK_ID"
# case $SLURM_ARRAY_TASK_ID in
#     1)
#         echo "udepth_rmi_sea_nerf"
#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#         --RMI --save_freq 1 --name udepth_rmi_Curasao \
#         --img_dir data/Seathru_nerf/Curasao --rescale_img

#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#         --RMI --save_freq 1 --name udepth_rmi_IUI3-RedSea \
#         --img_dir data/Seathru_nerf/IUI3-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#         --RMI --save_freq 1 --name udepth_rmi_JapaneseGradens-RedSea \
#         --img_dir data/Seathru_nerf/JapaneseGradens-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#         --RMI --save_freq 1 --name udepth_rmi_Panama \
#         --img_dir data/Seathru_nerf/Panama --rescale_img
#         ;;
#     2)
#         echo "udepth_rgb_sea_nerf"
#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#         --save_freq 1 --name udepth_rgb_Curasao \
#         --img_dir data/Seathru_nerf/Curasao --rescale_img

#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#         --save_freq 1 --name udepth_rgb_IUI3-RedSea \
#         --img_dir data/Seathru_nerf/IUI3-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#         --save_freq 1 --name udepth_rgb_JapaneseGradens-RedSea \
#         --img_dir data/Seathru_nerf/JapaneseGradens-RedSea --rescale_img

        # python evaluate_seaThru_correction.py --depth_model udepth \
        # --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
        # --save_freq 1 --name udepth_rgb_Panama \
        # --img_dir data/Seathru_nerf/Panama --rescale_img
#         ;;
#     3)
#         echo "dispnet_rgb_seathru_nerf"
#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rgb_Curasao \
#         --img_dir data/Seathru_nerf/Curasao --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rgb_IUI3-RedSea \
#         --img_dir data/Seathru_nerf/IUI3-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rgb_JapaneseGradens-RedSea \
#         --img_dir data/Seathru_nerf/JapaneseGradens-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rgb_Panama \
#         --img_dir data/Seathru_nerf/Panama --rescale_img
#         ;;
    

#     4)  echo "dispnet_rmi_seathru_nerf"
#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rmi_Curasao \
#         --img_dir data/Seathru_nerf/Curasao --RMI --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rmi_IUI3-RedSea \
#         --img_dir data/Seathru_nerf/IUI3-RedSea --RMI --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rmi_JapaneseGradens-RedSea \
#         --img_dir data/Seathru_nerf/JapaneseGradens-RedSea --RMI --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_rmi_Panama \
#         --img_dir data/Seathru_nerf/Panama --RMI --rescale_img
#         ;;
        
#     5)  
#         echo "dispnet mvit rgb sea_thru_nerf"
#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rgb_Curasao \
#         --img_dir data/Seathru_nerf/Curasao --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rgb_IUI3-RedSea \
#         --img_dir data/Seathru_nerf/IUI3-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rgb_JapaneseGradens-RedSea \
#         --img_dir data/Seathru_nerf/JapaneseGradens-RedSea --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rgb_Panama \
#         --img_dir data/Seathru_nerf/Panama --rescale_img
#         ;;
   
#     6)
#         echo "dispnet mvit rmi sea_thru_nerf"
        # python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
        # --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
        # --save_freq 1 --name dispnet_mvit_rmi_Curasao \
        # --img_dir data/Seathru_nerf/Curasao --RMI --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rmi_IUI3-RedSea \
#         --img_dir data/Seathru_nerf/IUI3-RedSea --RMI --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rmi_JapaneseGradens-RedSea \
#         --img_dir data/Seathru_nerf/JapaneseGradens-RedSea --RMI --rescale_img

#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 1 --name dispnet_mvit_rmi_Panama \
#         --img_dir data/Seathru_nerf/Panama --RMI --rescale_img       
#         ;;
# esac


# echo "Job $SLURM_ARRAY_TASK_ID"
# case $SLURM_ARRAY_TASK_ID in
#     1)
#         echo "udepth_rmi_varos"
#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#         --RMI --save_freq 10 --name udepth_rmi_varos \
#         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
#         ;;

#     2)
#         echo "udepth_rgb_varos"
#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#         --save_freq 10 --name udepth_rgb_varos \
#         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
#         ;;
#     3)
#         echo "dispnet_rgb_varos"
#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_rgb_varos \
#         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#         ;;
    

#     4)  echo "dispnet_rmi_varos"
#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_rmi_varos \
#         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
#         ;;
        
#     5)  
#         echo "dispnet_mvit_rgb_varos"
#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_mvit_rgb_varos \
#         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
#         ;;
   
#     6)
#         echo "dispnet_mvit_rmi_varos"
#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_mvit_rmi_varos \
#         --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img
#         ;;
# esac


# echo "Job $SLURM_ARRAY_TASK_ID"
# case $SLURM_ARRAY_TASK_ID in
#     1)
#         echo "udepth_rmi_eiffel"
        # python evaluate_seaThru_correction.py --depth_model udepth \
        # --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
        # --RMI --save_freq 10 --name udepth_rmi_eiffel \
#         ;;

#     2)
#         echo "udepth_rgb_eiffel"
#         python evaluate_seaThru_correction.py --depth_model udepth \
#         --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#         --save_freq 10 --name udepth_rgb_eiffel \
#         ;;
#     3)
#         echo "dispnet_rgb_eiffel"
#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_rgb_eiffel \
#         ;;
    

#     4)  echo "dispnet_rmi_eiffel"
#         python evaluate_seaThru_correction.py --depth_model dispnet \
#         --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_rmi_eiffel \
#         ;;
        
#     5)  
#         echo "dispnet_mvit_rgb_eiffel"
#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_mvit_rgb_eiffel \
#         ;;
    
#     6)
#         echo "dispnet_mvit_rmi_eiffel"
#         python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#         --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#         --save_freq 10 --name dispnet_mvit_rmi_eiffel \
#         ;;
# esac


# echo "Job $SLURM_ARRAY_TASK_ID"
#     case $SLURM_ARRAY_TASK_ID in
#         #eiffel
#         1)
#             echo "udepth_rmi_eiffel"
#             python evaluate_seaThru_correction.py --depth_model udepth \
#             --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#             --RMI --save_freq sq
#             1 --name udepth_rmi_eiffel
#             ;;
#             #checked
#         2)
#             echo "udepth_rgb_eiffel"
#             python evaluate_seaThru_correction.py --depth_model udepth \
#             --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#             --save_freq 1 --name udepth_rgb_eiffel
#             ;;
#             #checked
        
#         3)
#             echo "dispnet_mvit_rmi_eiffel"
#             python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#             --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#             --RMI --save_freq 1 --name dispnet_mvit_rmi_eiffel \
#             ;;
#             #checked
        
#         4)  
#             echo "dispnet_mvit_rgb_eiffel"
#             python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#             --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#             --save_freq 1 --name dispnet_mvit_rgb_eiffel \
#             ;;
#             #checked
    
#         5)  echo "dispnet_rmi_eiffel"
#             python evaluate_seaThru_correction.py --depth_model dispnet \
#             --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#             --RMI --save_freq 1 --name dispnet_rmi_eiffel \
#             ;;
#             #checked
#         6) 
#              echo "dispnet_rgb_eiffel"
#             python evaluate_seaThru_correction.py --depth_model dispnet \
#             --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#             --save_freq 1 --name dispnet_rgb_eiffel
#             ;;
#             #checked
#         #Varos

#         7)
#             echo "udepth_rmi_varos"
#             python evaluate_seaThru_correction.py --depth_model udepth \
#             --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#             --RMI --save_freq 1 --name udepth_rmi_varos \
#             --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#             ;;
#             #checked
#         8)
#             echo "udepth_rgb_varos"
#             python evaluate_seaThru_correction.py --depth_model udepth \
#             --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#             --save_freq 1 --name udepth_rgb_varos \
#             --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#             ;;
#             #checked
        
#         9)
#             echo "dispnet_mvit_rmi_varos"
#             python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#             --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#             --RMI --save_freq 1 --name dispnet_mvit_rmi_varos \
#             --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#             ;;
#             #checked
        
#         10)  
#             echo "dispnet_mvit_rgb_varos"
#             python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#             --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#             --save_freq 1 --name dispnet_mvit_rgb_varos \
#             --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#             ;;
#             #checked
    
#         11)  echo "dispnet_rmi_varos"
#             python evaluate_seaThru_correction.py --depth_model dispnet \
#             --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#             --RMI --save_freq 1 --name dispnet_rmi_varos \
#             --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#             ;;
#             #checked
#         12) 
#              echo "dispnet_rgb_varos"
#             python evaluate_seaThru_correction.py --depth_model dispnet \
#             --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#             --save_freq 1 --name dispnet_rgb_varos \
#             --img_dir /home/mundus/mrahman527/Varos/2021-08-17_SEQ1/vehicle0/cam0/A --rescale_img

#             ;;
#             #checked
#     esac    


# echo "Job $SLURM_ARRAY_TASK_ID"
#     case $SLURM_ARRAY_TASK_ID in
#         #Curasao

#         1)
#             echo "udepth_rmi_Curasao"
#             python evaluate_seaThru_correction.py --depth_model udepth \
#             --saved_model saved_models/udepth_actual_rmi_train/dispnet_model_best.pth.tar \
#             --RMI --save_freq 1 --name udepth_rmi_Curasao \
#             --img_dir data/Seathru_nerf/Curasao --rescale_img 
#             ;;
#             #checked
#         2)
#             echo "udepth_rgb_Curasao"
#             python evaluate_seaThru_correction.py --depth_model udepth \
#             --saved_model saved_models/udepth_rmi_train/dispnet_model_best.pth.tar \
#             --save_freq 1 --name udepth_rgb_Curasao \
#             --img_dir data/Seathru_nerf/Curasao --rescale_img
#             ;;
#             #checked
        
#         3)
#             echo "dispnet_mvit_rmi_Curasao"
#             python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#             --saved_model saved_models/dispnet_mvit_rmi/dispnet_model_best.pth.tar  \
#             --RMI --save_freq 1 --name dispnet_mvit_rmi_Curasao \
#             --img_dir data/Seathru_nerf/Curasao --rescale_img
#             ;;
#             #checked
        
#         4)  
#             echo "dispnet_mvit_rgb_Curasao"
#             python evaluate_seaThru_correction.py --depth_model dispnet_mvit \
#             --saved_model saved_models/dispnet_mvit_rgb/dispnet_model_best.pth.tar  \
#             --save_freq 1 --name dispnet_mvit_rgb_Curasao \
#             --img_dir data/Seathru_nerf/Curasao --rescale_img
#             ;;
#             #checked
    
#         5)  echo "dispnet_rmi_Curasao"
#             python evaluate_seaThru_correction.py --depth_model dispnet \
#             --saved_model saved_models/dispnet_rmi/dispnet_model_best.pth.tar  \
#             --RMI --save_freq 1 --name dispnet_rmi_Curasao \
#             --img_dir data/Seathru_nerf/Curasao --rescale_img
#             ;;
#             #checked
#         6) 
#             echo "dispnet_rgb_Curasao"
#             python evaluate_seaThru_correction.py --depth_model dispnet \
#             --saved_model saved_models/compare_new_train_without_mask/dispnet_model_best.pth.tar  \
#             --save_freq 1 --name dispnet_rgb_Curasao\
#             --img_dir data/Seathru_nerf/Curasao --rescale_img
#             ;;
#             #checked
#     esac    
