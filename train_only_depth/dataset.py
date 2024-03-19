import torch.utils.data as data
import numpy as np
from imageio.v2 import imread
from path import Path
import random
import os
from PIL import Image, ImageOps
# import matplotlib.pyplot as plt
import cv2

def load_as_float(path):
    return imread(path).astype(np.float32)

def load_depth(path):
    depth = cv2.imread(path)
    depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
    return depth

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
                    #world_T_tgt
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
                        #world_T_ref
                        ref_pose = np.array(pose_lines[i+j].split(' ')).astype(np.float32).reshape(3,4)
                        ref_pose = np.vstack((ref_pose, np.array([0,0,0,1])))
                        sample['ref_poses'].append(ref_pose)

                sequence_set.append(sample)
                
        random.shuffle(sequence_set)

        self.samples = sequence_set
    
    def _get_poses(self,tgt_pose, ref_pose):
        '''
        gets w_t_tgt and w_t_ref poses and returns ref_t_tgt
        ref_t_tgt = inv(w_t_ref) @ w_t_tgt
        '''
        assert tgt_pose.shape == (4, 4), "tgt_pose must have shape (4, 4)"
        assert ref_pose.shape == (4, 4), "ref_pose must have shape (4, 4)"
        
        return np.linalg.inv(ref_pose)@tgt_pose

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]

        if self.use_mask:
            tgt_mask = load_as_float(sample['tgt_mask'])
            ref_masks = [load_as_float(ref_mask) for ref_mask in sample['ref_masks']]

        else:
            tgt_mask = None
            ref_masks = []
        
        if self.use_depth:
            tgt_depth =  np.expand_dims(load_depth(sample['tgt_depth']),0)
            ref_depths = [np.expand_dims(load_depth(ref_depth),0) for ref_depth in sample['ref_depths']]
        else:
            tgt_depth = None
            ref_depths = []
        
        if self.use_pose:
            poses = [self._get_poses(sample['tgt_pose'], ref_pose) for ref_pose in sample['ref_poses']]
        else:
            poses = [] 
        # if self.transform is not None:
        #     imgs, intrinsics, masks = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']),[tgt_mask]+ref_masks, [tgt_depth]+ref_depths)
        #     tgt_img = imgs[0]
        #     ref_imgs = imgs[1:]
        #     tgt_mask = masks[0]
        #     ref_masks = masks[1:]
        data_sample = {
            'tgt_img': tgt_img,
            'ref_imgs': ref_imgs,
            'poses': poses,
            'inv_poses': [np.linalg.inv(pose) for pose in poses],
            'tgt_mask': tgt_mask,
            'ref_masks': ref_masks,
            'tgt_depth': tgt_depth,
            'ref_depths': ref_depths
        }
        data_sample['poses'] = rotMat2euler(data_sample['poses'])
        data_sample['inv_poses'] = rotMat2euler(data_sample['inv_poses'])
        
        data_sample['intrinsics'] = np.copy(sample['intrinsics'])
        data_sample['inv_intrinsics'] = np.linalg.inv(data_sample['intrinsics'])
        
        if self.transform is not None:
            data_sample = self.transform(data_sample)
        
        
        return data_sample

    def __len__(self):
        return len(self.samples)

from scipy.spatial.transform import Rotation  

def rotMat2euler(poses):
    '''
    takes a list of 4x4 poses and returns list of (1,6) poses euler angel
    '''
    poses_to_return = []

    for pose in poses:
        translation = pose[:3, 3]
        
        rotation_matrix = pose[:3, :3]
        r =  Rotation.from_matrix(rotation_matrix)
        angles = r.as_euler("xyz",degrees=False)
        
        pose_to_return = np.hstack((translation, angles))
        poses_to_return.append(pose_to_return)
        
    return poses_to_return

if __name__=='__main__':
    import custom_transforms

    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    transforms = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    dataset = SequenceFolderWithGT(
        root = '/mundus/mrahman527/Thesis/data/Eiffel_tower_ready_small_set',
        mask_root='/mundus/mrahman527/Thesis/data/scaled_and_cropped_mask', 
        depth_root='/mundus/mrahman527/Thesis/data/scaled_and_cropped_depth',
        seed=0, 
        train=True, 
        sequence_length=3, 
        transform=transforms, 
        skip_frames=1, 
        use_mask=False, 
        use_depth=True, 
        use_pose=False
    )

    data_sample = dataset[0]
    # for data in data_sample:
    #     print('---------',data,'---------')
    #     print(data_sample[data])
    # plt.imshow(data_sample['tgt_depth'].astype(np.uint8))
    # plt.colorbar()
    # plt.show()
    print(data_sample['tgt_img'].shape)
    print(data_sample['tgt_depth'].shape)
    print(data_sample['tgt_mask'])
    
    print(data_sample['ref_imgs'][0].shape)
    print(data_sample['ref_depths'][0].shape)
    print(data_sample['ref_masks'])


    print(data_sample['poses'])
    print(data_sample['intrinsics'].shape)
