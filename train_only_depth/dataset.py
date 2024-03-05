import torch.utils.data as data
import numpy as np
from imageio.v2 import imread
from path import Path
import random
import os
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
# import cv2

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolderWithGT(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/train.txt
        root/val.txt
        root/scene_1/0000000.png
        root/scene_1/0000001.png
        ..
        root/scene_1/cam.txt
        root/scene_1/gt_traj.txt 
        (containing real pose. inverse of the network output in Kitty format.(use function in utils.utils file))
        
        root/scene_2/0000000.png
        ...
        
        mask folder has structure:
            root/scene_1/mask_000.png
            root/scene_1/mask_001.png
            ...
            root/scene_2/mask_000.png
            ...
        depth folder has structure:
            root/scene_1/depth_000.png
            root/scene_1/depth_001.png
            ...
            root/scene_2/depth_000.png
            ...

       transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
        
    """

    def __init__(self, root, mask_root=None, depth_root=None ,seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, img_ext='png',use_mask=False, use_depth=False, use_pose=False):
        np.random.seed(seed)
        random.seed(seed)
        #check if the mask and depth root is given if they are to be used
        if use_depth:
            assert depth_root is not None
        if use_mask:
            assert mask_root is not None   

        
        self.use_mask = use_mask
        self.use_depth = use_depth
        self.use_pose = use_pose

        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scene_list_path = scene_list_path
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        
        if use_mask:
            self.mask_root = Path(mask_root)
            self.mask_scenes = [self.mask_root/folder[:-1] for folder in open(scene_list_path)]
            
        if use_depth:
            self.depth_root = Path(depth_root)
            self.depth_scenes = [self.depth_root/folder[:-1] for folder in open(scene_list_path)]

        #check if the number of depths and masks is same as the number of images
        self.img_ext = img_ext
        self.check_scenes()

        self.transform = transform
        self.k = skip_frames
        self.train = train
        self.crawl_folders(sequence_length)

    def check_scenes(self):
        '''
        check if number of files in depth, mask and number of poses are equal to the number of images
        '''
        for folder in open(self.scene_list_path):
            if self.use_mask:
                assert len((self.mask_root/folder[:-1]).files(f'*.{self.img_ext}'))==len((self.root/folder[:-1]).files(f'*.{self.img_ext}'))

            if self.use_depth:
                assert len((self.depth_root/folder[:-1]).files(f'*.{self.img_ext}'))==len((self.root/folder[:-1]).files(f'*.{self.img_ext}'))

            if self.use_pose:
                with open((self.root/folder[:-1]).files('gt_traj.txt')[0]) as f:
                    assert len(f.readlines())==len((self.root/folder[:-1]).files(f'*.{self.img_ext}'))


    def crawl_folders(self, sequence_length):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files(f'*.{self.img_ext}'))
            # print([img.split('/')[-1] for img in imgs][:10])
            if self.use_pose:
                with open(scene/'gt_traj.txt') as f:
                    pose_lines = f.readlines()

            if len(imgs) < sequence_length:
                continue
            
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                
                image_name = imgs[i].split('/')[-1]
                scene_name = imgs[i].split('/')[-2]
                
                if self.use_mask:
                    mask_name = self.mask_root/scene_name/('mask_'+image_name)
                else:
                    mask_name = None

                if self.use_depth:
                    depth_name = self.depth_root/scene_name/('depth_'+image_name)
                else:
                    depth_name = None
                
                if self.use_pose:
                    tgt_pose = np.array(pose_lines[i].split(' ')).astype(np.float32).reshape(3,4)
                    tgt_pose = np.vstack((tgt_pose, np.array([0,0,0,1])))
                else: 
                    tgt_pose=None

                sample = {'intrinsics': intrinsics, 'tgt': imgs[i],
                        'tgt_mask' : mask_name,'tgt_depth':depth_name,'tgt_pose':tgt_pose,
                         'ref_imgs': [], 'ref_masks':[], 'ref_depths':[], 'ref_poses':[]}
                
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                    image_name = imgs[i+j].split('/')[-1]
                    scene_name = imgs[i+j].split('/')[-2]
                    if self.use_mask:
                        mask_name = self.mask_root/scene_name/('mask_'+image_name)
                        sample['ref_masks'].append(mask_name)
                    
                    if self.use_depth:
                        depth_name = self.depth_root/scene_name/('depth_'+image_name)
                        sample['ref_depths'].append(depth_name)
                    
                    if self.use_pose:
                        ref_pose = np.array(pose_lines[i+j].split(' ')).astype(np.float32).reshape(3,4)
                        ref_pose = np.vstack((ref_pose, np.array([0,0,0,1])))
                        sample['ref_poses'].append(ref_pose)

                sequence_set.append(sample)
                
        random.shuffle(sequence_set)

        self.samples = sequence_set
    


    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        if self.use_mask:
            tgt_mask = load_as_float(sample['tgt_mask'])
            ref_masks = [load_as_float(ref_mask) for ref_mask in sample['ref_masks']]

        else:
            tgt_mask = None
            ref_masks = []
        
        if self.use_depth:
            tgt_depth = load_as_float(sample['tgt_depth'])
            ref_depths = [load_as_float(ref_mask) for ref_mask in sample['ref_masks']]
        else:
            tgt_depth = None
            ref_depths = []

        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        
        if self.transform is not None:
            imgs, intrinsics, masks = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']),[tgt_mask]+ref_masks)
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
            tgt_mask = masks[0]
            ref_masks = masks[1:]

        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics), tgt_mask, ref_masks

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    dataset = SequenceFolderWithGT(
        root = '/mundus/mrahman527/Thesis/data/Eiffel_tower_ready_small_set',
        mask_root='/mundus/mrahman527/Thesis/data/scaled_and_cropped_mask', 
        depth_root='/mundus/mrahman527/Thesis/data/scaled_and_cropped_depth',
        seed=0, 
        train=True, 
        sequence_length=3, 
        transform=None, 
        skip_frames=1, 
        use_mask=True, 
        use_depth=True, 
        use_pose=True
    )

    print(dataset[0])