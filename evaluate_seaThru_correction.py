import argparse

import timeit
import numpy as np
import os
from tqdm import tqdm
from path import Path
import torch
from imageio.v2 import imread, imsave
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import cv2
from datetime import datetime


#models
from models.SFM.DispNetS import DispNetS
from models.DepthAnything.DepthAnything import DepthAnythingSFM
from train_only_depth.udepth_model.udepth import UDepth_SFM

from utils.eval import median_of_non_zero_values


#sea_through
from train_only_depth.seathru import run_pipeline
#evaluate
from evaluate_image_recovery import rmetrics, nmetrics

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluate Depth")
parser.add_argument("--gt_depth", default='data/scaled_and_cropped_depth/2015', help="gt depth dir", type=str)
parser.add_argument("--img_dir", default='data/Eiffel-Tower_ready_Downscaled_colmap/2015',help="image directory for reading image", type=str)
parser.add_argument("--depth_model", default="dpts", help="model to select. options: dpts, udepth, dispnet", type=str)
parser.add_argument("--saved_model", default=None, type=str)
parser.add_argument("--RMI", action='store_true', help="Will use RMI input space for images")

parser.add_argument("--rescale_img", action="store_true", help='rescale images and depths to half the size')
parser.add_argument("--save_freq", default=0, type=int, help="save every this amount recovered image")
parser.add_argument("--name", required=True, help="name of the folder to be saved everything")
######################################################
class Args(argparse.Namespace):
    f = 2.0
    l = 0.5
    p = 0.01
    min_depth = 0.01
    max_depth = 1
    spread_data_fraction = 0.01
    # size = 320

def load_tensor_image(filename):

    img = imread(filename).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.5)/0.5)
    
    return tensor_img

def load_gt_depth(filename):
    gt_depth  = cv2.imread(filename)
    gt_depth = cv2.cvtColor(gt_depth, cv2.COLOR_BGR2GRAY)

    return gt_depth

def main(args):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'found device {device}')

    #check if valid directories
    image_dir = Path(args.img_dir)
    depth_dir = Path(args.gt_depth)
    assert os.path.exists(image_dir) and os.path.exists(depth_dir), 'image or depth directory invalid'
    image_list = sorted(image_dir.glob('*.png'))
    depth_list = sorted(depth_dir.glob('*.png'))

    assert len(image_list)==len(depth_list)

    #check correct model name given:
    assert args.depth_model in ['dpts', 'dptl', 'dptb', 'dispnet','udepth','dispnet_mvit'], "correct model name not given"

    #initialize the models
    if args.depth_model == 'dispnet':
        model = DispNetS(rmi=args.RMI)
        total_param = sum(param.numel() for param in model.parameters())
        print(f"total disp net parameter {total_param/1e6}M")
    
    elif args.depth_model == 'dpts':
        model = DepthAnythingSFM(encoder='vits')
    elif args.depth_model == 'dptl':
        model = DepthAnythingSFM(encoder='vitl')
    elif args.depth_model == 'dptb':
        model = DepthAnythingSFM(encoder='vitb')
    elif args.depth_model=='udepth':
        model = UDepth_SFM(load_pretrained=True, rmi=args.RMI)
    
    model.to(device)
    #load saved model
    if args.saved_model is not None:
        saved_model_path = Path(args.saved_model)
        print(f"loading {saved_model_path}")
        weights = torch.load(saved_model_path, map_location=device)
        
        if 'epoch' in weights:
            epoch = weights['epoch']
            print(f"loading model from epoch{weights['epoch']}")
        else:
            epoch = 0

        model.load_state_dict(weights['state_dict'], strict=False)
        model.eval()

    else:
        model.eval()
        print('No saved model given. Running with initialized parameters')
    
    save_path = "Image_enhancement_result"/Path(args.name)
    if os.path.exists(save_path):
        save_path = save_path+str(datetime.now())
    save_path.makedirs_p()

    sea_thru_args = Args()
    psnr_tot=0
    ssim_tot=0 
    uiqm_tot=0
    uciqe_tot=0
    for i,(image, depth) in tqdm(enumerate(zip(image_list, depth_list))):
        
        img = load_tensor_image(image).to(device)        
        gt_depth = load_gt_depth(depth)
        
        if args.rescale_img:
            img = torch.nn.functional.upsample(img, scale_factor=0.5, mode='bilinear')
            gt_depth = torch.nn.functional.upsample(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze().numpy()

        if args.depth_model == 'dispnet':
            pred_depth = model(img).cpu().squeeze().detach().numpy()
            pred_depth = 1/pred_depth

        else:
            pred_depth = model(img)[0].cpu().squeeze().detach().numpy()
            pred_depth = pred_depth

        img = img.detach().cpu().numpy().squeeze()
        img = np.moveaxis(img, 0, -1)


        recovered = run_pipeline(img*0.5+0.5, pred_depth, sea_thru_args)

        #convert to uint8 type
        recovered = (recovered*255).astype(np.uint8)
        img = (img*255).astype(np.uint8)
        
        psnr,ssim = rmetrics(recovered,img)
        uiqm,uciqe = nmetrics(recovered)
        
        psnr_tot+=psnr
        ssim_tot+=ssim
        uiqm_tot+=uiqm
        uciqe_tot+=uciqe

        if args.save_freq!=0 and i%args.save_freq==0:
            plt.imsave(save_path/('recovered_'+image.split('/')[-1]), recovered)

    

    return psnr_tot/(i+1), ssim_tot/(i+1), uiqm_tot/(i+1), uciqe_tot/(i+1)

        

    
    # return args.depth_model ,args.RMI, epoch, args.max_depth, abs_rel_tot/(len(image_list)-skipped), sq_rel_tot/(len(image_list)-skipped), log10_tot/(len(image_list)-skipped), rmse_tot/(len(image_list)-skipped), rmse_log_tot/(len(image_list)-skipped), a1_tot/(len(image_list)-skipped), a2_tot/(len(image_list)-skipped), a3_tot/(len(image_list)-skipped)


if __name__=='__main__':
    args = parser.parse_args()
    psnr, ssim, uiqm, uciqe = main(args)
    print('psnr=',psnr, 'ssim=',ssim, 'uiqm=',uiqm,'uicqe=', uciqe)