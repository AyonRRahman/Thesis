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

from skimage import exposure
from skimage.restoration import denoise_bilateral, denoise_tv_chambolle, estimate_sigma

#models
from models.SFM.DispNetS import DispNetS
from models.DepthAnything.DepthAnything import DepthAnythingSFM
from train_only_depth.udepth_model.udepth import UDepth_SFM

from utils.eval import median_of_non_zero_values
from csv import DictWriter

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
parser.add_argument("--use_gt_depth", action="store_true", help="use the ground truth depth for enhance image")
######################################################
class Args(argparse.Namespace):
    f = 2.0
    l = 0.5
    p = 0.01
    min_depth = 0.01
    max_depth = 0.4
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
    image_list = (image_dir.glob('*.png'))
    depth_list = sorted(depth_dir.glob('*.png'))

    # assert len(image_list)==len(depth_list)

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
    
    save_path = "Image_enhancement_result_all_image"/Path(args.name)
    if os.path.exists(save_path):
        save_path = save_path+str(datetime.now())
    save_path.makedirs_p()

    sea_thru_args = Args()
    psnr_tot=0
    ssim_tot=0 
    uiqm_tot=0
    uciqe_tot=0
    N = 0
    field_names = ['image', 'psnr', 'ssim', 'uiqm','uiqm_or', 'uciqe', 'uciqe_or']

    for i,(image, depth) in tqdm(enumerate(zip(image_list, depth_list))):
        
        img = load_tensor_image(image).to(device)        
        gt_depth = load_gt_depth(depth)
        
        if args.rescale_img:
            img = torch.nn.functional.upsample(img, scale_factor=0.75, mode='bilinear')
            gt_depth = torch.nn.functional.upsample(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0), scale_factor=0.75, mode='nearest').squeeze().numpy()

        if not args.use_gt_depth:
            if args.depth_model == 'dispnet':
                pred_depth = model(img).cpu().squeeze().detach().numpy()
                pred_depth = 1/pred_depth

            else:
            
                pred_depth = model(img)[0].cpu().squeeze().detach().numpy()
                pred_depth = pred_depth
        else:
            pred_depth=gt_depth
    
        # print(img.shape)
        img = img.detach().cpu().numpy().squeeze()
        img = np.moveaxis(img, 0, -1)
        img = exposure.equalize_adapthist(img*0.5 + 0.5, clip_limit=0.03)
        try:
            depths = pred_depth
            depths = ((depths - np.min(depths)) / (
                np.max(depths) - np.min(depths))).astype(np.float32)
            # depths = depths + 0.01
            mask = (depths<=0.4).astype(int)
            depths = mask*depths
            recovered = run_pipeline(img, depths, sea_thru_args)
            sigma_est = estimate_sigma(recovered, channel_axis=-1,average_sigmas=True) / 10.0
            recovered = denoise_tv_chambolle(recovered, sigma_est, channel_axis=-1)
            img_mask = np.stack([mask, mask, mask], axis=2)
            # print(img_mask.shape)
            
            recovered = img_mask*recovered + (1-img_mask)*img

        except Exception as e:
            print('-------')
            print(e)
            print(img.max(), img.min())
            print('-------')

            continue
        #convert to uint8 type
        recovered = (recovered*255).astype(np.uint8)
        img = (img*255).astype(np.uint8)
        try:
            psnr,ssim = rmetrics(recovered,img)
            
            uiqm,uciqe = nmetrics(recovered)
            uiqm_or, uciqe_or = nmetrics(img)
            print(f"{image.split('/')[-1]}: uiqm={uiqm}, uiqm_or={uiqm_or} uciqe={uciqe}, uciqe_or = {uciqe_or}")

            # print(uiqm, uiqm_or)
            # print(uciqe, uciqe_or)
        except Exception as e:
            print('-------')
            print(e)
            print('-------')

            
            continue
        
        N+=1        
        psnr_tot+=psnr
        ssim_tot+=ssim
        uiqm_tot+=uiqm
        uciqe_tot+=uciqe

        #result logging
        #uiqm or is the original images uiqm before recovering. same for uciqe
        dict = {'image':image.split('/')[-1], 'psnr':psnr, 'ssim':ssim, 'uiqm':uiqm,'uiqm_or':uiqm_or, 'uciqe':uciqe,'uciqe_or':uciqe_or}
        
        with open(save_path/'result.csv', 'a') as f_object:
            dictwriter_object = DictWriter(f_object, fieldnames=field_names)
            # Pass the dictionary as an argument to the Writerow()
            dictwriter_object.writerow(dict)
            # Close the file object
            f_object.close()


        if args.save_freq!=0 and i%args.save_freq==0:
            plt.imsave(save_path/('recovered_'+image.split('/')[-1]), recovered)

    
    with open(save_path/'result.txt', 'w') as f:
        f.write(f'psnr={psnr_tot/(N)}, ssim={ssim_tot/(N)}, uiqm={uiqm_tot/(N)}, uicqe={uciqe_tot/(N)}')
    print(f"total={i+1}, done={N}")
    return psnr_tot/(N), ssim_tot/(N), uiqm_tot/(N), uciqe_tot/(N)

        
@torch.no_grad
def get_mask_from_attn(range_attn_maps, threshold=0.8):
    # print(range_attn_maps.shape)
    attn = range_attn_maps.sum(1).detach()
    attn = (attn-attn.min())/(attn.max()-attn.min())
    # print(attn.shape)
    return attn>threshold

if __name__=='__main__':
    args = parser.parse_args()
    psnr, ssim, uiqm, uciqe = main(args)
    print('psnr=',psnr, 'ssim=',ssim, 'uiqm=',uiqm,'uicqe=', uciqe)