import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random
import os
from PIL import Image, ImageOps

def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, image_extention='png'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        # print(f"found these scenes: {self.scenes}")
        self.transform = transform
        # self.dataset = dataset
        self.k = skip_frames
        self.train = train
        self.crawl_folders(sequence_length, image_extention)

    def crawl_folders(self, sequence_length, image_extention):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files(f'*.{image_extention}'))

            if len(imgs) < sequence_length:
                continue
            
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        
        
        random.shuffle(sequence_set)

        self.samples = sequence_set

    def __getitem__(self, index):
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)




class SequenceFolderRMI(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .
        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, skip_frames=1, image_extention='png'):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        # print(f"found these scenes: {self.scenes}")
        self.transform = transform
        # self.dataset = dataset
        self.k = skip_frames
        self.train = train
        self.crawl_folders(sequence_length, image_extention)

    def crawl_folders(self, sequence_length, image_extention):
        # k skip frames
        sequence_set = []
        demi_length = (sequence_length-1)//2
        shifts = list(range(-demi_length * self.k, demi_length * self.k + 1, self.k))
        shifts.pop(demi_length)
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files(f'*.{image_extention}'))

            if len(imgs) < sequence_length:
                continue
            
            for i in range(demi_length * self.k, len(imgs)-demi_length * self.k):
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)
        
        
        random.shuffle(sequence_set)

        self.samples = sequence_set

    def _load_as_float(self, path):
        '''
        loads RMI input space
        '''
        image = imread(path).astype(np.uint8)
        r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
        # Compute R channel
        # r = np.array(r)
        # Compute M channel
        gb_max = np.maximum.reduce([g, b])
        # Compute I channel
        gray_c = np.array(ImageOps.grayscale(Image.fromarray(image)))
        # Combine three channels
        combined = np.stack((r, gb_max, gray_c), axis=-1)
        return combined.astype(np.float32)

    def __getitem__(self, index):
        sample = self.samples[index]
        print(sample['tgt'])
        tgt_img = self._load_as_float(sample['tgt'])
        ref_imgs = [self._load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)