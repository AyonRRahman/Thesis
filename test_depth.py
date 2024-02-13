import torch

from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm

from inverse_warp import pose_vec2mat
# from scipy.ndimage.interpolation import zoom

from inverse_warp import *

from models.SFM.DispNetS import DispNetS

from utils_SC import tensor2array

import os
import cv2
import numpy as np
from PIL import Image, ImageOps


parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained_dir", required=True, type=str, help="path containing pretrained PoseNet")
parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='depth_evaluation', type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png'], nargs='*', type=str, help="images extensions to glob")

parser.add_argument("--sequence", default='2015', type=str, help="sequence to test")
parser.add_argument("--use_best", action='store_true', help='use best model instead of last checkpoint.')
parser.add_argument("--name", type=str, help="folder to save the output in output-dir")
parser.add_argument("--use_RMI", action='store_true', help="use RMI input space")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# device = torch.device("cpu")
def load_as_RMI(filename):
    image = imread(filename).astype(np.uint8)
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    # Compute R channel
    # r = np.array(r)
    # Compute M channel
    gb_max = np.maximum.reduce([g, b])
    # Compute I channel
    gray_c = np.array(ImageOps.grayscale(Image.fromarray(image)))
    # Combine three channels
    combined = np.stack((r, gb_max, gray_c), axis=-1)
    return combined

def load_tensor_image(filename, args):
    if args.use_RMI:
        img = load_as_RMI(filename).astype(np.float32)
    else:
        img = imread(filename).astype(np.float32)
    
    h, w, _ = img.shape
    
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.5)/0.5).to(device)
    return tensor_img

import matplotlib.pyplot as plt
@torch.no_grad()
def main():
    args = parser.parse_args()

    if os.path.isfile(args.pretrained_dir):
        print(f"loading {args.pretrained_dir}")
        weights_pose = torch.load(args.pretrained_dir, map_location=device)
    else:
        if args.use_best:
            model_to_load = 'dispnet_model_best.pth.tar'
            
        else:
            model_to_load = 'dispnet_checkpoint.pth.tar'
         
        dispnet_saved_path = Path(args.pretrained_dir)/model_to_load
        print(f"loading {dispnet_saved_path}")
        weights_disp = torch.load(dispnet_saved_path, map_location=device)
        
    
    disp_net = DispNetS().to(device)
    
    print(f'loading model from epoch{weights_disp['epoch']}')
    disp_net.load_state_dict(weights_disp['state_dict'], strict=False)
    disp_net.eval()

    image_dir = Path(args.dataset_dir)/args.sequence
    output_dir = Path(args.output_dir)/args.name/args.sequence
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))
    n = len(test_files)
    
    if args.use_RMI:
        print('loading as RMI input space')
        
    for iter in tqdm(range(n )):

        tensor_img = load_tensor_image(test_files[iter], args)

        # print(test_files[iter])
        # print(f"img shape {tensor_img.shape}")

        disp = disp_net(tensor_img)[0,0]
        depth = 1/disp
        # print(depth.shape)
        # plt.imshow(depth.detach().cpu())
        # plt.savefig('a.png')
        depth = depth.detach().cpu().numpy().astype(np.float64)
        # print(depth.shape)
        
        # print(type(depth))
        # print(f'dep {depth.max()}')
        img_gray = cv2.cvtColor((depth*255.0).astype(np.uint8),cv2.COLOR_GRAY2BGR)
        # print(img_gray.shape)
        
        # print(f'gray {np.array(img_gray).max()}')
        # cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        filename = output_dir/test_files[iter].split('/')[-1]

        cv2.imwrite(filename, img_gray)
        print(iter)
        # print(f'disp shape {disp.shape}')
        # print(f'depth shape {depth.shape}')
        # print(filename)
        




if __name__ == '__main__':
    main()