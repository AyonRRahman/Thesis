
import argparse
from path import Path
import os
import time
import csv
import datetime
from tqdm import tqdm
import cv2
from imageio import imread
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import custom_transforms
from utils.eval import mean_squared_error_depth
from utils_SC import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder, SequenceFolderRMI,SequenceFolderMask

from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='Structure from Motion Learner training on Eiffel-Tower Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('mask_dir',metavar='DIR', help='Path to the masks', default='./data/scaled_and_cropped_mask')
# parser.add_argument('--folder-type', type=str, choices=['sequence'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=16, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
# parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
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
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')


parser.add_argument('--optimize_loss_weight', action='store_true', help='optimize the weightages for different loss')
parser.add_argument('--depth_directory', type=str, default='/mundus/mrahman527/Thesis/data/Eiffel-Tower_depth_images/', help='depth images directory')
parser.add_argument('--use_pretrained', action='store_true', help='to start from the last saved models')
parser.add_argument('--use_RMI', action='store_true', help='use RMI input space instead of RGB.')
parser.add_argument('--train_until_converge', action='store_true', help='train till converging without stopping')
parser.add_argument('--use_mask_for_train', action='store_true',
            help='new dataloader with mask will be used and the mask will be used to calculate the loss')

parser.add_argument("--depth_model", default="dispnet", help="model to select. options: dpts, dptl, dptb, dispnet", type=str)

val_depth_iter = 0
best_error = -1
n_iter = 0
n_iter_without_best = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    #print gpu information
    num_gpus = torch.cuda.device_count()

    # Loop through each GPU and print its properties
    for gpu_id in range(num_gpus):
        gpu_props = torch.cuda.get_device_properties(gpu_id)
        print(f"GPU {gpu_id}:")
        print(f"  Name: {gpu_props.name}")
        print(f"  Total Memory: {gpu_props.total_memory / 1024**3:.2f} GB")

w1, w2, w3 = None, None, None


def load_as_float(path):
    return imread(path).astype(np.float32)

def sigmoid(x):
    #not sigmoid now. Exp to make it softmax
    return torch.exp(x)

def main():
    global best_error, n_iter, device, n_iter_without_best
    global w1, w2, w3

    args = parser.parse_args()

    if args.use_mask_for_train and not os.path.exists(args.mask_dir):
        print('got argument to use mask for training but the provided mask dir does not exists')
        return

    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight
    print(f'initial weights: {w1, w2, w3}')
    print(f"batch size {args.batch_size}")

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    if args.train_until_converge:
        args.epochs=4000
        print('setting max epochs to 4000 as train untill converge')
    if not args.use_pretrained:
        args.save_path = 'checkpoints'/save_path/timestamp
    else:
        args.save_path = 'checkpoints'/save_path
    
    args.model_save_path = 'saved_models'/save_path
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()
    
    args.continue_training = False

    '''
    if the model save path exists and not to use pretrained, rename with timestamp
    elif the model save path exists and use pretrained, load the models from this directory
    else make the new directory to save models
    '''

    #where models will get saved
    if os.path.exists(args.model_save_path) and not args.use_pretrained:
        args.model_save_path = args.model_save_path + timestamp
        args.model_save_path.makedirs_p()

    elif os.path.exists(args.model_save_path) and args.use_pretrained:
        args.continue_training = True

    else:
        args.model_save_path.makedirs_p()

    print(f'models will be saved in {args.model_save_path}')

    
    
    
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

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    
    global val_set # to use it in the validation step to get the gt depth


    depth_val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                image_extention = args.image_extention
            )

    if not args.use_mask_for_train:
        if not args.use_RMI:
            print('not using RMI input space')
            train_set = SequenceFolder(
                args.data,
                transform=train_transform,
                seed=args.seed,
                train=True,
                sequence_length=args.sequence_length,
                image_extention = args.image_extention
            )
            #Validation set is the same type as training set to measure photometric loss from warping

            # global val_set # to use it in the validation step to get the gt depth

            val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                image_extention = args.image_extention
            )
        else:
            print('using RMI input space')
            train_set = SequenceFolderRMI(
                args.data,
                transform=train_transform,
                seed=args.seed,
                train=True,
                sequence_length=args.sequence_length,
                image_extention = args.image_extention
            )
            #Validation set is the same type as training set to measure photometric loss from warping

            
            val_set = SequenceFolderRMI(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                image_extention = args.image_extention
            )
    else:
        print('Using Masked Dataset')
        train_set = SequenceFolderMask(
            args.data,
            args.mask_dir,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            image_extention = args.image_extention
        )
           
        val_set = SequenceFolderMask(
            args.data,
            args.mask_dir,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            image_extention = args.image_extention
        )

        # depth_val_set = SequenceFolder(
        #         args.data,
        #         transform=valid_transform,
        #         seed=args.seed,
        #         train=False,
        #         sequence_length=args.sequence_length,
        #         image_extention = args.image_extention
        #     )


    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    depth_val_loader = torch.utils.data.DataLoader(
        depth_val_set, batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    
    #custom 
    from models.SFM.DispNetS import DispNetS
    from models.DepthAnything.DepthAnything import DepthAnythingSFM
    # from models.Udepth.udepth import UDepth_SFM
    from models.udepth_model.udepth import UDepth_SFM
    if args.depth_model=='dispnet':
        disp_net = DispNetS().to(device)
    elif args.depth_model=='dpts':
        # print('loading DepthAnything model')
        # disp_net = DepthAnythingSFM(encoder='vits').to(device)
        print('loading udepth')
        disp_net = UDepth_SFM(True)

    from models.SC_SFM.PoseResNet import PoseResNet
    pose_net = PoseResNet(18, args.with_pretrain).to(device)
    epoch_skip = 0
    # load parameters
    if args.continue_training:
        #load the best performing models
        print(f'Continuing Training from the models saved in {args.model_save_path}')
        args.pretrained_pose = Path(args.model_save_path)/'exp_pose_model_best.pth.tar'
        args.pretrained_disp = Path(args.model_save_path)/'dispnet_model_best.pth.tar'
        if not os.path.exists(args.pretrained_disp):
            print('disp net best saved model not found in the save directory. starting traiing from scratch')
            args.pretrained_disp = None
        
        if not os.path.exists(args.pretrained_pose):
            print('pose net best saved model not found in the save directory. starting traiing from scratch')
            args.pretrained_pose = None
        
        
    if args.pretrained_disp:
        print(f"=> using pre-trained weights for DispNet from given path {args.pretrained_disp}")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)
        epoch_skip = weights['epoch']-1

    if args.pretrained_pose:
        print(f"=> using pre-trained weights for PoseResNet from given path {args.pretrained_pose}")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)
        
        #load the saved weights if optimize_loss_weight is true
        if args.optimize_loss_weight and ('w1' in weights):
            print('=> loading weights')
            w1 = weights['w1']
            w2 = weights['w2']
            w3 = weights['w3']
    
    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    if not args.optimize_loss_weight:
        optim_params = [
            {'params': disp_net.parameters(), 'lr': args.lr},
            {'params': pose_net.parameters(), 'lr': args.lr}
        ]
    else:
        #converting the weights to tensor and then adding them to the params list
        w1, w2, w3 = torch.tensor(w1, dtype=torch.float64, requires_grad=True), torch.tensor(w2,dtype=torch.float64, requires_grad=True), torch.tensor(w3,dtype=torch.float64, requires_grad=True)
        # w1.requires_grad = True
        # w2.requires_grad = True
        # w3.requires_grad = True
        w1.to(device)
        w2.to(device)
        w3.to(device)

        optim_params = [
            {'params': disp_net.parameters(), 'lr': args.lr},
            {'params': pose_net.parameters(), 'lr': args.lr},
            {'params': w1, 'lr': args.lr},
            {'params': w2, 'lr': args.lr},
            {'params': w3, 'lr': args.lr}
        ]

    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

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

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)
        
        if args.continue_training and epoch_skip!=0:
            if epoch<=epoch_skip:
                n_iter+= len(train_loader)
                continue

        # train for one epoch
        logger.reset_train_bar()
        if not args.use_mask_for_train:
            train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        else:
            train_loss = train_with_mask(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

        # evaluate on validation set
        logger.reset_valid_bar()
        
        if not args.use_mask_for_train:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt_with_mask(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)

        #mse on depth with gt
        print(f'validating depth {epoch}')
        
        # if not args.use_mask_for_train:
        mse = validate_depth(depth_val_loader, disp_net, training_writer, args)        
        training_writer.add_scalar('mse depth', mse, epoch)
        
        # else:
        #     #need to write this part of the script    
        #     pass

        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

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
            if n_iter_without_best==80:
                print(f"model is not converging for last 80 epoch. stoping at {epoch}")
                break
        best_error = min(best_error, decisive_error)

        #args.model_save_path
        if not args.optimize_loss_weight:
            save_checkpoint(
                args.model_save_path, {
                    'epoch': epoch + 1,
                    'state_dict': disp_net.module.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.module.state_dict()
                },
                is_best)
        else:
            save_checkpoint(
                args.model_save_path, {
                    'epoch': epoch + 1,
                    'state_dict': disp_net.module.state_dict(),
                    'w1': w1.item(),
                    'w2': w2.item(),
                    'w3': w3.item()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': pose_net.module.state_dict(),
                    'w1': w1.item(),
                    'w2': w2.item(),
                    'w3': w3.item()
                },
                is_best)

        
        with torch.no_grad():
            if args.optimize_loss_weight:  #print the weights
                print(f"Epoch: {epoch}, w1: {(sigmoid(w1)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))).item()}, w2: {(sigmoid(w2)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))).item()}, w3: {(sigmoid(w3)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))).item()}")
        
        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()

def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
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

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        # print(f'depth output shapes {tgt_depth[0].shape}')
        # print(tgt_depth.shape)
        # print(ref_depths[0].shape)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        if not args.optimize_loss_weight:
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        else:
            loss = (sigmoid(w1)*loss_1 + sigmoid(w2)*loss_2 + sigmoid(w3)*loss_3)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))

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



def train_with_mask(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    print('training with mask')
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

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, tgt_mask, ref_masks) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        tgt_mask = tgt_mask.to(device)
        ref_masks = [mask.to(device) for mask in ref_masks]
        intrinsics = intrinsics.to(device)

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs, args)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode, tgt_mask, ref_masks)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        if not args.optimize_loss_weight:
            loss = w1*loss_1 + w2*loss_2 + w3*loss_3
        else:
            loss = (sigmoid(w1)*loss_1 + sigmoid(w2)*loss_2 + sigmoid(w3)*loss_3)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))

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
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    #need to add the gt depth images in tensorboard 
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
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)
        # if i==1:
        #     break
        # compute output
        if args.depth_model=='dispnet':
            tgt_depth = [1/disp for disp in disp_net(tgt_img)]
        elif args.depth_model=='dpts':
            tgt_depth = [disp for disp in disp_net(tgt_img)]

        if args.depth_model=='dispnet':
            tgt_depth = [1 / disp_net(tgt_img)]
            # print('validating in dispnet')
            # print(tgt_depth[0][0].shape)
            ref_depths = []
            for ref_img in ref_imgs:
                ref_depth = [1 / disp_net(ref_img)]
                ref_depths.append(ref_depth)
            
            # print(ref_depths[0][0].shape)

        elif args.depth_model=='dpts':
            tgt_depth = disp_net(tgt_img)
            ref_depths = []

            for ref_img in ref_imgs:
                ref_depth = disp_net(ref_img)
                ref_depths.append(ref_depth)



        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                current_image = (val_set.samples[i])['tgt'].split('/')[-1]
                depth_file_name = Path(args.depth_directory)/current_image[0:4]/'depth_images'/'depth_'+current_image
                #try to use the tensor2array function later for depth
                print('------------')
                # print(f'current image name {(val_set.samples[i])['tgt'].split('/')[-1]}')
                print(f"depth_file_name {depth_file_name}")
                print('------------')
                
                depth_image = load_as_float(depth_file_name)
                depth_image = depth_image/depth_image.max()
                output_writers[i].add_image('gt depth', tensor2array(torch.tensor(depth_image), max_value=None, colormap='magma'), 0)
                

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        
        #making sure to stop calculating the weights gradients.
        with torch.no_grad():
            if not args.optimize_loss_weight:
                loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            else:
                loss = (sigmoid(w1)*loss_1 + sigmoid(w2)*loss_2 + sigmoid(w3)*loss_3)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))

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
def validate_without_gt_with_mask(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    print('validating with mask')
    #need to add the gt depth images in tensorboard 
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
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv, tgt_mask, ref_masks) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        tgt_mask = tgt_mask.to(device)
        ref_masks = [mask.to(device) for mask in ref_masks]
        
        # compute output
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                current_image = (val_set.samples[i])['tgt'].split('/')[-1]
                depth_file_name = Path(args.depth_directory)/current_image[0:4]/'depth_images'/'depth_'+current_image
                #try to use the tensor2array function later for depth
                print('------------')
                # print(f'current image name {(val_set.samples[i])['tgt'].split('/')[-1]}')
                print(f"depth_file_name {depth_file_name}")
                print('------------')
                
                depth_image = load_as_float(depth_file_name)
                depth_image = depth_image/depth_image.max()
                output_writers[i].add_image('gt depth', tensor2array(torch.tensor(depth_image), max_value=None, colormap='magma'), 0)
                

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode, tgt_mask, ref_masks)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        
        #making sure to stop calculating the weights gradients.
        with torch.no_grad():
            if not args.optimize_loss_weight:
                loss = w1*loss_1 + w2*loss_2 + w3*loss_3
            else:
                loss = (sigmoid(w1)*loss_1 + sigmoid(w2)*loss_2 + sigmoid(w3)*loss_3)/(sigmoid(w1)+sigmoid(w2)+sigmoid(w3))

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
def load_gt_depth(current_image):
    year = current_image[0:4]
    depth_dir = Path('data/scaled_and_cropped_depth/')/year
    depth_img = depth_dir/('depth_'+current_image)
    depth_gt = cv2.imread(depth_img).astype(np.uint8)
    depth_gt_gray = cv2.cvtColor(depth_gt, cv2.COLOR_BGR2GRAY)
    return depth_gt_gray

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
        # if i==1:
        #     break
        if args.depth_model=='dispnet':
            tgt_depth = 1 / disp_net(tgt_img)
            predicted_depth = tgt_depth.squeeze().detach().cpu().numpy()
        
        elif args.depth_model=='dpts':
            tgt_depth = disp_net(tgt_img)
            predicted_depth = tgt_depth[0].squeeze().detach().cpu().numpy()

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
        # print(tgt_img.shape)
        tgt_depth = [disp for disp in disp_net(tgt_img)]

    # print(len(tgt_depth))

        ref_depths = []
        for ref_img in ref_imgs:
            # print(ref_img.shape)
            ref_depth = [disp for disp in disp_net(ref_img)]
            ref_depths.append(ref_depth)

    return tgt_depth, ref_depths

if __name__ == '__main__':
    main()
