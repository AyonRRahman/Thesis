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

from models.SFM.DispNetS import DispNetS
from models.SFM.DispNetS_mvit import DispNet_mvit
from models.DepthAnything.DepthAnything import DepthAnythingSFM
from train_only_depth.udepth_model.udepth import UDepth_SFM

from utils.eval import median_of_non_zero_values

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluate Depth")
parser.add_argument("--img_dir", default='data/Eiffel-Tower_ready_Downscaled_colmap/2015',help="image directory for reading image", type=str)
parser.add_argument("--depth_model", default="dpts", help="model to select. options: dpts, udepth, dispnet", type=str)
parser.add_argument("--saved_model", default=None, type=str)
parser.add_argument("--RMI", action='store_true', help="Will use RMI input space for images")
parser.add_argument("--rescale_img", action="store_true", help='rescale images and depths to half the size(for Varos)')
parser.add_argument("--name", type=str, required=True, help="name of the folder to save the prediction")
######################################################

def load_tensor_image(filename, use_RMI=False):

    img = imread(filename).astype(np.float32)
    
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.5)/0.5)
    
    return tensor_img


def main():
    args = parser.parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'found device {device}')

    #check if valid directories
    image_dir = Path(args.img_dir)
    assert os.path.exists(image_dir), 'image directory invalid'

    image_list = sorted(image_dir.glob('*.png'))
    
    #check correct model name given:
    assert args.depth_model in ['dispnet','udepth','dispnet_mvit'], "correct model name not given"

    #initialize the models
    if args.depth_model == 'dispnet':
        model = DispNetS(rmi=args.RMI)
        print('loading dispnet')
    elif args.depth_model=='dispnet_mvit':
        print('loading dispnet mvit')
        model = DispNet_mvit(rmi=args.RMI).to(device)
    elif args.depth_model=='udepth':
        print('loading udepth')
        model = UDepth_SFM(load_pretrained=True, rmi=args.RMI)

    total_param = sum(param.numel() for param in model.parameters())
    print(f"total network parameter {total_param/1e6}M")

    model.to(device)
    #load saved model
    if args.saved_model is not None:
        saved_model_path = Path(args.saved_model)
        print(f"loading {saved_model_path}")
        weights = torch.load(saved_model_path, map_location=device)
        
        if 'epoch' in weights:
            epoch = weights['epoch']
            print(f"loading model from epoch{weights['epoch']}")

        model.load_state_dict(weights['state_dict'], strict=False)
        model.eval()

    else:
        print('No saved model given.')
        exit()
    
    save_dir = Path('Predicted_Depth')/args.name
    save_dir.makedirs_p()
    for i,(image) in tqdm(enumerate(image_list)):
        
        img = load_tensor_image(image).to(device)        
       
        if args.rescale_img:
            img = torch.nn.functional.upsample(img, scale_factor=0.5, mode='bilinear')

        if args.depth_model == 'dispnet':
            pred_depth = model(img).cpu().squeeze().detach().numpy()
            pred_depth = 1/pred_depth
        else:
            pred_depth = model(img)[0].cpu().squeeze().detach().numpy()
            pred_depth = pred_depth

        file_name = save_dir/'depth_'+image.split('/')[-1]
        # print(pred_depth.max())
        cv2.imwrite(file_name, (255*(pred_depth)).astype(np.uint8))
        
        # if i==10:
        #     break
       

        

    

if __name__=="__main__":
    main()
