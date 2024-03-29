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


from models.SC_SFM.PoseResNet import PoseResNet
from utils_SC import tensor2array

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
from utils.eval import align_trajectory, find_ATE

parser = argparse.ArgumentParser(description='Script for getting odometry',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained_dir", required=True, type=str, help="path containing pretrained PoseNet")


parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", default='pose_evaluation', type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)

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


@torch.no_grad()
def main():
    args = parser.parse_args()

    if os.path.isfile(args.pretrained_dir):
        print(f"loading {args.pretrained_dir}")
        weights_pose = torch.load(args.pretrained_dir, map_location=device)
    else:
        if args.use_best:
            model_to_load = 'exp_pose_model_best.pth.tar'
        else:
            model_to_load = 'exp_pose_checkpoint.pth.tar'
         
        posenet_saved_path = Path(args.pretrained_dir)/model_to_load
        print(f"loading {posenet_saved_path}")
        weights_pose = torch.load(posenet_saved_path, map_location=device)
        
    pose_net = PoseResNet().to(device)
    print(f'loading model from epoch{weights_pose['epoch']}')
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()

    image_dir = Path(args.dataset_dir)/args.sequence
    output_dir = Path(args.output_dir)/args.name
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()

    print('{} files to test'.format(len(test_files)))
    
    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]

    n = len(test_files)
    tensor_img1 = load_tensor_image(test_files[0], args)
    if args.use_RMI:
        print('loading as RMI input space')
        
    for iter in tqdm(range(n - 1)):

        tensor_img2 = load_tensor_image(test_files[iter+1], args)

        pose = pose_net(tensor_img1, tensor_img2)

        pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
        pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
        # print(pose_mat[3,0:3])
        # print(pose_mat[0:3, 0:3].T)
        # inverse_of_pose = np.vstack([np.hstack([pose_mat[0:3, 0:3].T, -(pose_mat[0:3, 0:3]@(pose_mat[3,0:3].reshape(3,1))) ]), np.array([0, 0, 0, 1])])

        # print(inverse_of_pose - np.linalg.inv(pose_mat))

        # global_pose = global_pose @ inverse_of_pose
        
        global_pose = global_pose @  np.linalg.inv(pose_mat)

        poses.append(global_pose[0:3, :].reshape(1, 12))

        # update
        tensor_img1 = tensor_img2

    poses = np.concatenate(poses, axis=0)
    if not args.use_best:
        filename = output_dir/(args.sequence+".txt")
    else:
        filename = output_dir/(args.sequence+"_best.txt")

    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')

    #calculating the ATE using umeyama alignment
    predicted_pose_file = filename
    gt_pose_file = f"data/Eiffel-Tower/{args.sequence}/gt_traj.txt"

    transformed_pred, ult_gt = align_trajectory(predicted_pose_file, gt_pose_file)

    print(f"total and mean ATE {find_ATE(transformed_pred, ult_gt)} m")



if __name__ == '__main__':
    main()