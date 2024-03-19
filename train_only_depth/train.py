#import libraries
import custom_transforms
from dataset import SequenceFolderWithGT

import sys
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '<your available GPU IDs>' 
import argparse
import datetime
import time
from path import Path
import csv
from tqdm import tqdm
#torch related imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data


import numpy as np
from tensorboardX import SummaryWriter

#setting the path to the project root for facillitating importing
script_path = os.path.abspath(__file__)
project_folder = os.path.abspath(os.path.join(os.path.dirname(script_path), '..'))
sys.path[0] = project_folder

#project imports
from logger import TermLogger, AverageMeter
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from datasets.sequence_folders import SequenceFolder
from utils.eval import mean_squared_error_depth
from utils_SC import tensor2array, save_checkpoint
import custom_transforms as custom_SC


### arg parser
parser = argparse.ArgumentParser(description='Structure from Motion Learner training on Eiffel-Tower Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='DIR', default='/mundus/mrahman527/Thesis/data/Eiffel_tower_ready_small_set',help="dataset directory")
parser.add_argument('--mask_dir', help='Path to the masks', default='/mundus/mrahman527/Thesis/data/scaled_and_cropped_mask', type=str)
parser.add_argument('--depth_dir', help='Path to the depths', default='/mundus/mrahman527/Thesis/data/scaled_and_cropped_depth', type=str)

parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=5, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
# parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')

parser.add_argument('--image_extention', type=str, default='png', help='image extention')
parser.add_argument('--name', dest='name',default='experiment', type=str,
                     required=True, 
                     help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')

parser.add_argument('--optimize_loss_weight', action='store_true', help='optimize the weightages for different loss')
parser.add_argument('--use_pretrained', action='store_true', help='to start from the last saved models')
parser.add_argument('--use_RMI', action='store_true', help='use RMI input space instead of RGB.')
parser.add_argument('--train_until_converge', action='store_true', help='train till converging without stopping')

parser.add_argument("--depth_model", default="dispnet", help="model to select. options: dpts, dispnet", type=str)

parser.add_argument('--use_gt_mask', action='store_true',
            help='use mask in the training process')
        
parser.add_argument('--use_gt_depth', action='store_true',
            help='use gt_depth in the training process')
        
parser.add_argument('--use_gt_pose', action='store_true',
            help='use gt_pose in the training process')

parser.add_argument('--train', default="both", choices=['depth', 'pose','both'], 
                    help='depth: train only depth network  with gt pose, pose: train only pose with gt depth', type=str)

parser.add_argument('--manual_weight', action='store_true',
            help='manual weight initialize')


#define global variables
val_depth_iter = 0
best_error = -1
n_iter = 0
n_iter_without_best = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

def log_gradients_in_model(model, writer, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Log the gradient as a histogram
            writer.add_histogram(f"gradients/{name}", param.grad.data, step)
        else:
            print(f"found grad None for tag={tag}, value={value}")


# def init_weights(self):
#     for m in self.modules():
#         if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#             xavier_uniform_(m.weight)
#             if m.bias is not None:
#                 zeros_(m.bias)
def init_weights(m):
    # if isinstance(m, nn.Linear):
    try:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    except Exception as e: 
        print(e)

#print gpu information if gpu is available
if torch.cuda.is_available():
    #print gpu information
    num_gpus = torch.cuda.device_count()

    # Loop through each GPU and print its properties
    for gpu_id in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        print(f"GPU {gpu_id}:")
        print(f"  Name: {gpu_props.name}")
        print(f"  Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")

def main():
    global best_error, n_iter, device, n_iter_without_best
    global w1, w2, w3

    args = parser.parse_args()
    print("====="*10)
    if args.use_gt_mask:
        print('Will use gt mask for training')
    
    if args.use_gt_depth:
        print('Will use gt depth for training')

    if args.use_gt_pose:
        print('Will use gt pose for training')

    #check if the dataset directories are existing
    assert os.path.exists(args.data), f"given dataset directory {args.data} does not exists"
    assert os.path.exists(args.mask_dir), f"given mask directory {args.mask_dir} does not exists"
    assert os.path.exists(args.depth_dir), f"given depth directory {args.depth_dir} does not exists"

    #check if train mode and the using gt data is consistent
    if args.train=="pose":
        assert args.use_gt_depth, "ground truth depth is needed for training only pose"
    elif args.train=="depth":
        assert args.use_gt_pose, "ground truth pose is needed for training only depth"
    
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    print("====="*10)
    print(f'initial weights: {w1, w2, w3}')
    print(f"batch size {args.batch_size}")
    print("====="*10)

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)

    #check for train untill converge to set max epoch
    if args.train_until_converge:
        args.epochs=4000
        print('setting max epochs to 4000 as train until converge')
    
    #set the save path
    #if the save path already exists and not to continue training from existing savepoint
    #rename the save path with timestamp
    if not args.use_pretrained and os.path.exists('checkpoints_experiment'/save_path):
        args.save_path = 'checkpoints_experiment'/save_path/timestamp
    else:
        args.save_path = 'checkpoints_experiment'/save_path
    
    args.model_save_path = 'saved_models_experiment'/save_path
    if os.path.exists(args.model_save_path) and not args.use_pretrained:
        args.model_save_path = args.model_save_path/timestamp
    
    print('=> will save everything to {}'.format(args.save_path))
    print('=> will save models to {}'.format(args.model_save_path))

    #check condition for continuing training from saved model
    args.continue_training = False
    
    if os.path.exists(args.model_save_path) and args.use_pretrained:
        args.continue_training = True

    #make the save directories
    if not os.path.exists(args.save_path):
        args.save_path.makedirs_p()

    if not os.path.exists(args.model_save_path):
        args.model_save_path.makedirs_p()

    #set the seeds for random functions
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    #tensorbord writer
    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    #validation output image tensorboard writer
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    #setting datasets and dataloaders
    #transforms
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])

    train_transforms = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transforms = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    #Train and Validation set
    train_set = SequenceFolderWithGT(
        root = args.data,
        mask_root=args.mask_dir, 
        depth_root=args.depth_dir,
        seed=args.seed, 
        train=True, 
        sequence_length=args.sequence_length, 
        transform=train_transforms, 
        skip_frames=1, 
        use_mask=args.use_gt_mask, 
        use_depth=args.use_gt_depth, 
        use_pose=args.use_gt_pose
    )
    global val_set
    val_set = SequenceFolderWithGT(
        root = args.data,
        mask_root=args.mask_dir, 
        depth_root=args.depth_dir,
        seed=args.seed, 
        train=False, 
        sequence_length=args.sequence_length, 
        transform=valid_transforms,  
        use_mask=args.use_gt_mask, 
        use_depth=args.use_gt_depth, 
        use_pose=args.use_gt_pose
        )
    
    depth_valid_transform = custom_SC.Compose([custom_transforms.ArrayToTensor(), normalize])

    depth_val_set = SequenceFolder(
        args.data,
        transform=valid_transforms,
        seed=args.seed,
        train=False,
        sequence_length=args.sequence_length,
        image_extention = args.image_extention
        )

    print("====="*10)
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    print("====="*10)
    
    #Data Loader
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    depth_val_loader = torch.utils.data.DataLoader(
        depth_val_set, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    #set epoch size if not given
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    print("=> creating model")
    
    #import models
    from models.SC_SFM.PoseResNet import PoseResNet 
    from models.SFM.DispNetS import DispNetS
    from models.DepthAnything.DepthAnything import DepthAnythingSFM

    if args.depth_model == "dispnet":
        disp_net = DispNetS().to(device)

    elif args.depth_model =="dpts":
        disp_net = DepthAnythingSFM(encoder='vits').to(device)

    pose_net = PoseResNet(18, args.with_pretrain).to(device)

    #epochs to skip if continue training from saved model
    epoch_skip = 0

    #handle pretrained network weights paths before loading them
    if args.continue_training:
        #load the best performing models
        print(f'Continuing Training from the models saved in {args.model_save_path}')

        #giving priority to given posenet or dispnet model path over the model saved in the directory
        if args.pretrained_pose is None:
            args.pretrained_pose = Path(args.model_save_path)/'exp_pose_model_best.pth.tar'
        if args.pretrained_disp is None:
            args.pretrained_disp = Path(args.model_save_path)/'dispnet_model_best.pth.tar'

        if not os.path.exists(args.pretrained_disp):
            print(f'disp net best saved model not found in the save directory {args.pretrained_disp}. starting traiing from scratch')
            args.pretrained_disp = None
        
        if not os.path.exists(args.pretrained_pose):
            print(f'pose net best saved model not found in the save directory {args.pretrained_pose}. starting traiing from scratch')
            args.pretrained_pose = None
    
    
    if args.manual_weight:
        print("====="*10)
        print('manually initializing weights for dispnet')
        disp_net.apply(init_weights)
        print("====="*10)

    
    #load the pretrained models
    if args.pretrained_disp:
        print(f"=> using pre-trained weights for DispNet from given path {args.pretrained_disp}")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
        epoch_skip = weights['epoch']-1

    if args.pretrained_pose:
        print(f"=> using pre-trained weights for PoseResNet from given path {args.pretrained_pose}")
        weights = torch.load(args.pretrained_pose,map_location=device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
        #skip the smallest number of epoch in case different saved models from different run is given
        if epoch_skip>weights['epoch']-1:
            epoch_skip = weights['epoch']-1
        
    #parallel training in multigpu if available
    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')

    #set parameters based on the training mode and del other model to free memory
    if args.train=='both':
        optim_params = [
            {'params': disp_net.parameters(), 'lr': args.lr},
            {'params': pose_net.parameters(), 'lr': args.lr}
        ]

    elif args.train=='depth':
        optim_params = [
            {'params': disp_net.parameters(), 'lr': args.lr}
        ]
        #load a saved pose net to use it in the training. gt pose not working
        #pose_net = None
        print('loading saved model for pose net ')
        weights = torch.load('/mundus/mrahman527/Thesis/saved_models/compare_new_train_with_mask/exp_pose_model_best.pth.tar',map_location=device)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
        pose_net = torch.nn.DataParallel(pose_net)


    elif args.train=='pose':
        optim_params = [
            {'params': pose_net.parameters(), 'lr': args.lr}
        ]
        depth_net = None


    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    #if the log files exists will append on it otherwise create it for continuation of training
    print("====="*10)
    if not os.path.exists(args.save_path/args.log_summary):
        with open(args.save_path/args.log_summary, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'validation_loss'])

    if not os.path.exists(args.save_path/args.log_full):    
        with open(args.save_path/args.log_full, 'w') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    # print('-----'*10)
    # print('named parameters for disp net')
    # print(list(disp_net.named_parameters()))
    # print('-----'*10)
    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)
        
        if args.continue_training and epoch_skip!=0:
            if epoch<=epoch_skip:
                n_iter+= len(train_loader)
                continue
        
        logger.reset_train_bar()
        # print(args)
        if args.train=='depth':
            train_loss = train_depth(args, train_loader, disp_net,pose_net, optimizer, args.epoch_size, logger, training_writer)

        elif args.train=='pose':
            train_loss = train_pose(args, train_loader, pose_net, optimizer, args.epoch_size, logger, training_writer)
        
        elif args.train=='both':
            train_loss = train_both(args, train_loader, disp_net,pose_net, optimizer, args.epoch_size, logger, training_writer)
            

        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()

        #validation
        if args.train=='depth':
            errors, error_names = validate_train_depth(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        
        elif args.train=='pose':
            errors, error_names = validate_train_pose(args, val_loader, pose_net, epoch, logger, output_writers)
            pass

        elif args.train=='both':
            errors, error_names = validate_train_both(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
            

        #mse on depth with gt
        print(f'validating depth after epoch: {epoch+1}')
        
        # if args.train=='depth':
        #     mse = validate_depth(depth_val_loader, disp_net, training_writer, args)        
        #     training_writer.add_scalar('mse depth', mse, epoch)

        # elif args.train=='pose':
        #     pass

        # elif args.train=='both':
        #     mse = validate_depth(depth_val_loader, disp_net, training_writer, args)        
        #     training_writer.add_scalar('mse depth', mse, epoch)
        #     pass

        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        #validation results logging in the tensorboard
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1] #0 = total weighted loss, 1 =photo loss
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        if is_best:
            n_iter_without_best=0
        else:
            n_iter_without_best+=1
            if n_iter_without_best==400:
                print(f"model is not converging for last 80 epoch. stoping at {epoch}")
                break
        best_error = min(best_error, decisive_error)

        #args.model_save_path
        if args.train=='both':
            save_checkpoint(
                args.model_save_path, {
                    'epoch': epoch + 1,
                    'state_dict': disp_net.module.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.module.state_dict()
                },
                is_best)
        elif args.train=='depth':
            save_checkpoint(
                args.model_save_path, {
                    'epoch': epoch + 1,
                    'state_dict': disp_net.module.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': disp_net.module.state_dict()
                },
                is_best)
        
        elif args.train=='pose':
            save_checkpoint(
                args.model_save_path, {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.module.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.module.state_dict()
                },
                is_best)
            
        
        
        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])

    logger.epoch_bar.finish()


def train_both(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    global w1, w2, w3
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    
    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    
    for i,data_sample in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        tgt_img = data_sample['tgt_img'].to(device)
        ref_imgs = [img.to(device) for img in data_sample['ref_imgs']]
        intrinsics = data_sample['intrinsics'].to(device)
        intrinsics_inv = data_sample['inv_intrinsics'].to(device)
        
        # poses = [pose.to(device) for pose in data_sample['poses']]  #src_t_tgt
        # poses_inv = [inv_pose.to(device) for inv_pose in data_sample['inv_poses']]

        tgt_mask = data_sample['tgt_mask'].to(device)
        ref_masks = [mask.to(device) for mask in data_sample['ref_masks']]
        

        
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        
        
        if args.use_gt_mask:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode, tgt_mask, ref_masks)

        else:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        
        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= (epoch_size - 1):
            break

        n_iter += 1
        
        if i%100==0:
            # Check how much GPU memory is currently being used
            memory_used = torch.cuda.memory_allocated()
            print(f"Current GPU memory used: {memory_used / 1024**3:.2f} GB")

            # Check how much GPU memory is currently cached
            memory_cached = torch.cuda.memory_cached()
            print(f"Current GPU memory cached: {memory_cached / 1024**3:.2f} GB")
            print(f"Max GPU memory used: {torch.cuda.max_memory_allocated()/1024**3} GB")
        
    return losses.avg[0]


def train_depth(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    global w1, w2, w3
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    
    # switch to train mode
    disp_net.train()

    #swith pose net to eval mode
    pose_net.eval()

    end = time.time()
    logger.train_bar.update(0)

    
    for i,data_sample in enumerate(train_loader):
        
        log_losses = i > 0 and n_iter % args.print_freq == 0
        # if i==1:
        #     pass
        # else:
        #     log_losses = True

        # measure data loading time
        data_time.update(time.time() - end)
        
        tgt_img = data_sample['tgt_img'].to(device)
        ref_imgs = [img.to(device) for img in data_sample['ref_imgs']]
        intrinsics = data_sample['intrinsics'].to(device)
        intrinsics_inv = data_sample['inv_intrinsics'].to(device)
        
        # poses = [pose.to(device) for pose in data_sample['poses']]  #src_t_tgt
        # poses_inv = [inv_pose.to(device) for inv_pose in data_sample['inv_poses']]

        tgt_mask = data_sample['tgt_mask'].to(device)
        ref_masks = [mask.to(device) for mask in data_sample['ref_masks']]
        

        
        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        
        with torch.no_grad():
            poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        
        
        if args.use_gt_mask:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode, tgt_mask, ref_masks)

        else:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        
        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #log gradients
        log_gradients_in_model(disp_net, train_writer, n_iter)


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= (epoch_size - 1):
            break

        n_iter += 1
        
        if i%100==0:
            # Check how much GPU memory is currently being used
            memory_used = torch.cuda.memory_allocated()
            print(f"Current GPU memory used: {memory_used / 1024**3:.2f} GB")

            # Check how much GPU memory is currently cached
            memory_cached = torch.cuda.memory_cached()
            print(f"Current GPU memory cached: {memory_cached / 1024**3:.2f} GB")
            print(f"Max GPU memory used: {torch.cuda.max_memory_allocated()/1024**3} GB")



    return losses.avg[0]

def train_pose(args, train_loader, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    global w1, w2, w3
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    
    # switch to train mode
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    
    for i,data_sample in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0
        
        # measure data loading time
        data_time.update(time.time() - end)
        
        tgt_img = data_sample['tgt_img'].to(device)
        ref_imgs = [img.to(device) for img in data_sample['ref_imgs']]
        intrinsics = data_sample['intrinsics'].to(device)
        intrinsics_inv = data_sample['inv_intrinsics'].to(device)
        
        # poses = [pose.to(device) for pose in data_sample['poses']]  #src_t_tgt
        # poses_inv = [inv_pose.to(device) for inv_pose in data_sample['inv_poses']]

        tgt_mask = data_sample['tgt_mask'].to(device)
        ref_masks = [mask.to(device) for mask in data_sample['ref_masks']]
        
        tgt_depth_d = data_sample['tgt_depth'].to(device)
        ref_depths_d = [depth.to(device) for depth in data_sample['ref_depths']]
        
        
        tgt_depth = [tgt_depth_d]
        
        ref_depths = []
        
        for ref_depth in ref_depths_d:   
            ref_depths.append([ref_depth])
        

        
        # compute output
        # tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        
        #compute losses
        if args.use_gt_mask:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, 1, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode, tgt_mask, ref_masks)

        else:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, 1, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)
        

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        
        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= (epoch_size - 1):
            break

        n_iter += 1
        
        if i%100==0:
            # Check how much GPU memory is currently being used
            memory_used = torch.cuda.memory_allocated()
            print(f"Current GPU memory used: {memory_used / 1024**3:.2f} GB")

            # Check how much GPU memory is currently cached
            memory_cached = torch.cuda.memory_cached()
            print(f"Current GPU memory cached: {memory_cached / 1024**3:.2f} GB")
            print(f"Max GPU memory used: {torch.cuda.max_memory_allocated()/1024**3} GB")
        
    return losses.avg[0]

@torch.no_grad()
def validate_train_pose(args, val_loader, pose_net, epoch, logger, output_writers):
    global device
    global val_set
    global w1, w2, w3

    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    pose_net.eval()
    end = time.time()
    logger.valid_bar.update(0)
    
    for i,data_sample in enumerate(val_loader):
        tgt_img = data_sample['tgt_img'].to(device)
        ref_imgs = [img.to(device) for img in data_sample['ref_imgs']]
        intrinsics = data_sample['intrinsics'].to(device)
        intrinsics_inv = data_sample['inv_intrinsics'].to(device)
        
        # poses = [pose.to(device) for pose in data_sample['poses']]  #src_t_tgt
        # poses_inv = [inv_pose.to(device) for inv_pose in data_sample['inv_poses']]

        tgt_mask = data_sample['tgt_mask'].to(device)
        ref_masks = [mask.to(device) for mask in data_sample['ref_masks']]
        
        tgt_depth_d = data_sample['tgt_depth'].to(device)
        ref_depths_d = [depth.to(device) for depth in data_sample['ref_depths']]
        
        
        tgt_depth = [tgt_depth_d]
        
        ref_depths = []
        
        for ref_depth in ref_depths_d:   
            ref_depths.append([ref_depth])
        
        

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                current_image = (val_set.samples[i])['tgt'].split('/')[-1]
                depth_file_name = Path(args.depth_dir)/current_image[0:4]/'depth_'+current_image
                #try to use the tensor2array function later for depth
                print('------------')
                # print(f'current image name {(val_set.samples[i])['tgt'].split('/')[-1]}')
                print(f"depth_file_name {depth_file_name}")
                print('------------')
                
                depth_image = load_as_float(depth_file_name)
                depth_image = np.transpose(depth_image, (2, 0, 1))
                # print(depth_image.shape)
                depth_image = depth_image/depth_image.max()
                output_writers[i].add_image('gt depth', tensor2array(torch.tensor(depth_image), max_value=None, colormap='magma'), 0)
                

            # output_writers[i].add_image('val Dispnet Output Normalized',
            #                             tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
            #                             epoch)
            # output_writers[i].add_image('val Depth Output',
            #                             tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
            #                             epoch)
        # compute output
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        if args.use_gt_mask:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode, tgt_mask, ref_masks)
        else:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)
            

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        
        #making sure to stop calculating the weights gradients.
        with torch.no_grad():
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            
        # loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

        
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']


@torch.no_grad()
def validate_train_depth(args, val_loader, disp_net, pose_net, epoch, logger, output_writers):
    global device
    global val_set
    global w1, w2, w3

    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0
    print(f'len of output writers {len(output_writers)}')

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i,data_sample in enumerate(val_loader):
        # if i==1:
        #     break
        tgt_img = data_sample['tgt_img'].to(device)
        ref_imgs = [img.to(device) for img in data_sample['ref_imgs']]
        intrinsics = data_sample['intrinsics'].to(device)
        intrinsics_inv = data_sample['inv_intrinsics'].to(device)
        
        # poses = [pose.to(device) for pose in data_sample['poses']]  #src_t_tgt
        # poses_inv = [inv_pose.to(device) for inv_pose in data_sample['inv_poses']]

        tgt_mask = data_sample['tgt_mask'].to(device)
        ref_masks = [mask.to(device) for mask in data_sample['ref_masks']]
        
        # compute output
        # tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        with torch.no_grad():
            poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                current_image = (val_set.samples[i])['tgt'].split('/')[-1]
                depth_file_name = Path(args.depth_dir)/current_image[0:4]/'depth_'+current_image
                #try to use the tensor2array function later for depth
                print('------------')
                # print(f'current image name {(val_set.samples[i])['tgt'].split('/')[-1]}')
                print(f"depth_file_name {depth_file_name}")
                print('------------')
                
                depth_image = load_as_float(depth_file_name)
                depth_image = np.transpose(depth_image, (2, 0, 1))
                print(depth_image.shape)
                depth_image = depth_image/depth_image.max()
                output_writers[i].add_image('gt depth', tensor2array(torch.tensor(depth_image), max_value=None, colormap='magma'), 0)
                

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)

        # poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        if args.use_gt_mask:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode, tgt_mask, ref_masks)
        else:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)
            

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        
        #making sure to stop calculating the weights gradients.
        with torch.no_grad():
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            
        # loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

        if i>=(args.epoch_size-1):
            break

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']


@torch.no_grad()
def validate_train_both(args, val_loader, disp_net, pose_net, epoch, logger, output_writers):
    global device
    global val_set
    global w1, w2, w3

    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i,data_sample in enumerate(val_loader):
        tgt_img = data_sample['tgt_img'].to(device)
        ref_imgs = [img.to(device) for img in data_sample['ref_imgs']]
        intrinsics = data_sample['intrinsics'].to(device)
        intrinsics_inv = data_sample['inv_intrinsics'].to(device)
        
        # poses = [pose.to(device) for pose in data_sample['poses']]  #src_t_tgt
        # poses_inv = [inv_pose.to(device) for inv_pose in data_sample['inv_poses']]

        tgt_mask = data_sample['tgt_mask'].to(device)
        ref_masks = [mask.to(device) for mask in data_sample['ref_masks']]
        
        # compute output
        # tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)
        
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                current_image = (val_set.samples[i])['tgt'].split('/')[-1]
                depth_file_name = Path(args.depth_dir)/current_image[0:4]/'depth_'+current_image
                #try to use the tensor2array function later for depth
                print('------------')
                # print(f'current image name {(val_set.samples[i])['tgt'].split('/')[-1]}')
                print(f"depth_file_name {depth_file_name}")
                print('------------')
                
                depth_image = load_as_float(depth_file_name)
                depth_image = np.transpose(depth_image, (2, 0, 1))
                print(depth_image.shape)
                depth_image = depth_image/depth_image.max()
                output_writers[i].add_image('gt depth', tensor2array(torch.tensor(depth_image), max_value=None, colormap='magma'), 0)
                

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)

        # poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        if args.use_gt_mask:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode, tgt_mask, ref_masks)
        else:
            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)
            

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        
        #making sure to stop calculating the weights gradients.
        with torch.no_grad():
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            
        # loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

        
    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']



@torch.no_grad()
def validate_depth(depth_val_loader, disp_net, training_writer, args):
    global device
    global val_set
    global val_depth_iter
    # switch to evaluate mode
    disp_net.eval()
    total_mse_loss = 0
    
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in tqdm(enumerate(depth_val_loader)):
        tgt_img = tgt_img.to(device)
        # compute output
        if args.depth_model=='dispnet':
            tgt_depth = 1 / disp_net(tgt_img)
            predicted_depth = tgt_depth.squeeze().detach().cpu().numpy()
        
        elif args.depth_model=='dpts':
            tgt_depth = disp_net(tgt_img)
            predicted_depth = tgt_depth.squeeze().detach().cpu().numpy()

        current_image = (val_set.samples[i])['tgt'].split('/')[-1]
        gt_depth = load_gt_depth(current_image)

        mse, scale = mean_squared_error_depth(predicted=predicted_depth, gt=gt_depth, normalize=False)
        if scale is not None:
            training_writer.add_scalar('scale', scale, val_depth_iter)
            val_depth_iter+=1

        total_mse_loss += mse
        
    return total_mse_loss/len(depth_val_loader)


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv

def compute_depth(disp_net, tgt_img, ref_imgs, args):
    # print(f"tgt_depth shape {disp_net(tgt_img).shape}")
    if args.depth_model=='dispnet':
        tgt_depth = [1/disp for disp in disp_net(tgt_img)]
        
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1/disp for disp in disp_net(ref_img)]
            ref_depths.append(ref_depth)

    elif args.depth_model=='dpts':
        tgt_depth = [disp for disp in disp_net(tgt_img)]


        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [disp for disp in disp_net(ref_img)]
            ref_depths.append(ref_depth)

    return tgt_depth, ref_depths

from imageio.v2 import imread

def load_as_float(path):
    return imread(path).astype(np.float32)


if __name__=='__main__':
    main()