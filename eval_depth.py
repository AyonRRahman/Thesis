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
from models.DepthAnything.DepthAnything import DepthAnythingSFM
from train_only_depth.udepth_model.udepth import UDepth_SFM

from utils.eval import median_of_non_zero_values

################### Options ######################
parser = argparse.ArgumentParser(description="Evaluate Depth")
parser.add_argument("--gt_depth", default='data/scaled_and_cropped_depth/2015', help="gt depth dir", type=str)
parser.add_argument("--img_dir", default='data/Eiffel-Tower_ready_Downscaled_colmap/2015',help="image directory for reading image", type=str)
parser.add_argument("--scale", action='store_true', help="Will correct the scale using Median")
parser.add_argument("--depth_model", default="dpts", help="model to select. options: dpts, udepth, dispnet", type=str)
parser.add_argument("--saved_model", default=None, type=str)
parser.add_argument("--RMI", action='store_true', help="Will use RMI input space for images")
parser.add_argument("--max_depth", default=0, type=float, help="maximum depth to mask the gt depth")
parser.add_argument("--calculate_inference_time", action='store_true', help="calculate the avg inference time of the model")
parser.add_argument("--save_file_name", default="depth_eval_result.csv",help="file name for saving the results")
parser.add_argument("--rescale_img", action="store_true", help='rescale images and depths to half the size')
######################################################


def predict_fn(model, input_data):

    return model(input_data)

def compute_depth_errors(gt, pred, use_mask=True, max_depth=0, scale=True):
    """Computation of error metrics between predicted and ground truth depths
    Args:
        gt (N): ground truth depth
        pred (N): predicted depth
    """
    if max_depth!=0:
        mask_max_depth = (gt<=max_depth).astype(int)
        pred = pred*mask_max_depth
        gt = gt*mask_max_depth

    if use_mask:
        mask = np.array(gt, dtype=bool).astype(int)
        pred = (pred*mask) #element wise multiply the mask
    
    #flatten and get the non zeros to calculate rest to avoid zero division
    flat_gt = gt.flatten()
    gt = flat_gt[flat_gt != 0]
    
    flat_pred = pred.flatten()
    pred = flat_pred[flat_pred != 0]

    #handling the situation where no gt depth is under the max depth
    if pred.shape != gt.shape or pred.size==0:
        # print(pred.shape, gt.shape)
        return None, None, None, None, None, None, None, None
    
    if scale:
        gt_median = median_of_non_zero_values(gt)
        predicted_median = median_of_non_zero_values(pred)
        scale = gt_median/predicted_median
        pred = pred*scale

    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    
    
    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs((np.log10(gt) - np.log10(pred))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel,log10 , rmse, rmse_log, a1, a2, a3

def load_tensor_image(filename, use_RMI=False):

    img = imread(filename).astype(np.float32)
    # print(f'img load shape {img.shape}')
    if use_RMI:
        img = img.astype(np.uint8)
        r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # Compute R channel
        # r = np.array(r)
        # Compute M channel
        gb_max = np.maximum.reduce([g, b])
        # Compute I channel
        gray_c = np.array(ImageOps.grayscale(Image.fromarray(img)))
        # Combine three channels
        combined = np.stack((r, gb_max, gray_c), axis=-1)
        img = combined.astype(np.float32)

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
        
    abs_rel_tot = 0
    sq_rel_tot = 0
    rmse_tot = 0
    rmse_log_tot = 0
    a1_tot = 0 
    a2_tot = 0
    a3_tot = 0
    log10_tot = 0
    skipped = 0
    for i,(image, depth) in tqdm(enumerate(zip(image_list, depth_list))):
        
        img = load_tensor_image(image).to(device)        
        gt_depth = load_gt_depth(depth)
        if args.rescale_img:
            # print(img.shape)
            # print(gt_depth.shape)
            img = torch.nn.functional.upsample(img, scale_factor=0.5, mode='bilinear')
            gt_depth = torch.nn.functional.upsample(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(0), scale_factor=0.5, mode='nearest').squeeze().numpy()
            # print(img.shape)
            # print(gt_depth.shape)
        if args.depth_model == 'dispnet':
            pred_depth = model(img).cpu().squeeze().detach().numpy()
        else:
            pred_depth = model(img)[0].cpu().squeeze().detach().numpy()

        if args.depth_model == 'dispnet':
            pred_depth = 1/pred_depth
        else:
            pred_depth = pred_depth


        if i==0 and args.calculate_inference_time:
            print('calculating model avg inference time')
            number_of_iterations = 100  # Number of times to run the prediction

            # Time the prediction function
            time_taken = timeit.timeit(lambda: predict_fn(model, img), number=number_of_iterations)

            # Calculate average prediction time per iteration
            average_prediction_time = time_taken / number_of_iterations

            print(f"avg prediction time: {average_prediction_time}")

        # if args.scale:
        #     gt_median = median_of_non_zero_values(gt_depth)
        #     predicted_median = median_of_non_zero_values(pred_depth)
        #     scale = gt_median/predicted_median
        #     # print(scale)
            
        #     pred_depth = pred_depth*scale


        abs_rel, sq_rel, log10, rmse, rmse_log, a1, a2, a3 = compute_depth_errors(gt_depth, pred_depth, max_depth=args.max_depth, scale=args.scale)
        
        if abs_rel is None:
            skipped+=1
            continue

        abs_rel_tot += abs_rel
        sq_rel_tot += sq_rel
        rmse_tot += rmse
        rmse_log_tot += rmse_log
        a1_tot += a1
        a2_tot += a2
        a3_tot += a3
        log10_tot += log10
    
    print(f'abs_rel: {abs_rel}')
    print(f'total skipped {skipped}')
    return args.depth_model ,args.RMI, epoch, args.max_depth, abs_rel_tot/(len(image_list)-skipped), sq_rel_tot/(len(image_list)-skipped), log10_tot/(len(image_list)-skipped), rmse_tot/(len(image_list)-skipped), rmse_log_tot/(len(image_list)-skipped), a1_tot/(len(image_list)-skipped), a2_tot/(len(image_list)-skipped), a3_tot/(len(image_list)-skipped)

if __name__=='__main__':
    from csv import writer
    args = parser.parse_args()
    
    csv_file = args.save_file_name
    
    if not os.path.exists(csv_file):
        row = ['model', 'RMI', 'epoch','max_depth', 'abs_rel_tot', 'sq_rel_tot', 'log_10_tot', 'rmse_tot', 'rmse_log_tot', 'a1_tot', 'a2_tot', 'a3_tot']
        with open(csv_file, 'w') as f_object:
            writer_object = writer(f_object)
 
            
            # the writerow()
            writer_object.writerow(row)
        
            # Close the file object
            f_object.close()

    model, RMI, epoch, max_depth, abs_rel_tot, sq_rel_tot, log_10_tot, rmse_tot, rmse_log_tot, a1_tot, a2_tot, a3_tot = main(args)
    
    print(f"result: abs_rel={abs_rel_tot}, sq_rel={sq_rel_tot},log_10={log_10_tot}, rmse={rmse_tot}, rmse_log ={rmse_log_tot}, a1 = {a1_tot}, a2 = {a2_tot}, a3 = {a3_tot}")

    with open(csv_file, 'a') as f_object:
        row = [model, RMI, epoch, max_depth, abs_rel_tot, sq_rel_tot, log_10_tot, rmse_tot, rmse_log_tot, a1_tot, a2_tot, a3_tot]
        writer_object = writer(f_object)
        # the writerow()
        writer_object.writerow(row)
        # Close the file object
        f_object.close()

