from __future__ import division
import torch
import random
import numpy as np
from PIL import Image

'''Set of tranform random routines that takes list of inputs as arguments,
in order to have random but coherent transformations.'''


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)

        return sample

class ArrayToTensor(object):
    """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

    def __call__(self, sample):
        sample_to_return = {}
        for items in sample:
            if items=='tgt_img':
                # put it from HWC to CHW format
                im = np.transpose(sample[items], (2, 0, 1))
                #convert to tensor
                sample_to_return[items] = torch.from_numpy(im).float()/255
            
            elif items=='ref_imgs':
                tensor_ref_imgs = []
                for im in sample[items]:
                    im = np.transpose(im, (2,0,1))
                    tensor_ref_imgs.append(torch.from_numpy(im).float()/255)

                sample_to_return[items]= tensor_ref_imgs
            
            elif items=='tgt_mask' or items=='tgt_depth':
                if  sample[items] is not None:
                    sample_to_return[items] = torch.from_numpy(sample[items]).float()
                else: 
                    sample_to_return[items]=[]

            elif items=='ref_masks' or items=='ref_depths':
                sample_to_return[items] = [torch.from_numpy(mask).float() for mask in sample[items]]
            
            elif items=='intrinsics':
                sample_to_return[items] = torch.from_numpy(sample[items]).float()
            
            elif items=='inv_intrinsics':
                sample_to_return[items] = torch.from_numpy(sample[items]).float()
            
            elif items in ['poses', 'inv_poses']:
                sample_to_return[items] = [torch.from_numpy(pose).float() for pose in sample[items]]
        
        return sample_to_return
        
        
class Normalize(object):
    '''
    normalizes the images with the mean and the std given
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        
        for t, m, s in zip(sample['tgt_img'], self.mean, self.std):
            t.sub_(m).div_(s)

        for tensors in sample['ref_imgs']:
            for t, m, s in zip(tensors, self.mean, self.std):
                t.sub_(m).div_(s)
        
        return sample

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, images, intrinsics, masks=None):
#         for t in self.transforms:
#             images, intrinsics, masks = t(images, intrinsics, masks)
#         if masks==None:
#             return images, intrinsics
#         else:
#             return images, intrinsics, masks


# class Normalize(object):
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, images, intrinsics, masks=None):
#         for tensor in images:
#             for t, m, s in zip(tensor, self.mean, self.std):
#                 t.sub_(m).div_(s)
#         # print(f"normalize {masks.shape}")
#         return images, intrinsics, masks


# class ArrayToTensor(object):
#     """Converts a list of numpy.ndarray (H x W x C) along with a intrinsics matrix to a list of torch.FloatTensor of shape (C x H x W) with a intrinsics tensor."""

#     def __call__(self, images, intrinsics, masks=None):
#         tensors = []
#         for im in images:
#             # put it from HWC to CHW format
#             im = np.transpose(im, (2, 0, 1))
#             # handle numpy array
#             tensors.append(torch.from_numpy(im).float()/255)
        
#         if masks is not None:
#             # masks_tensor = torch.from_numpy(masks).int()
#             masks_tensor = []
#             for mask in masks:
#                 # print(f"array to tensor {mask.shape}")
#                 masks_tensor.append(torch.from_numpy(mask).int())
            
#         else:
#             masks_tensor = None

#         # print(f"array2tensor {masks_tensor[0].shape}")

#         return tensors, intrinsics, masks_tensor


# class RandomHorizontalFlip(object):
#     """Randomly horizontally flips the given numpy array with a probability of 0.5"""

#     def __call__(self, images, intrinsics, masks=None):
#         assert intrinsics is not None
#         # print(f"masks type {type(masks)}")
        
#         if random.random() < 0.5:
#             output_intrinsics = np.copy(intrinsics)
#             output_images = [np.copy(np.fliplr(im)) for im in images]
#             w = output_images[0].shape[1]
#             output_intrinsics[0, 2] = w - output_intrinsics[0, 2]
            
#             if masks is None:
#                 output_masks = None
#             else:
#                 # print(f"flip {masks.shape}")
#                 output_masks = [np.copy(np.fliplr(mask)) for mask in masks] 

#         else:
#             output_images = images
#             output_intrinsics = intrinsics
#             output_masks= masks
        
#         # print(f"masks type {type(output_masks)}")
#         # print(f"random flip {output_masks[0].shape}")

#         return output_images, output_intrinsics, output_masks


# class RandomScaleCrop(object):
#     """Randomly zooms images up to 15% and crop them to keep same size as before."""

#     def __call__(self, images, intrinsics, masks=None):
#         assert intrinsics is not None
#         output_intrinsics = np.copy(intrinsics)

#         in_h, in_w, _ = images[0].shape
#         x_scaling, y_scaling = np.random.uniform(1, 1.15, 2)
#         scaled_h, scaled_w = int(in_h * y_scaling), int(in_w * x_scaling)

#         output_intrinsics[0] *= x_scaling
#         output_intrinsics[1] *= y_scaling
#         scaled_images = [np.array(Image.fromarray(im.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for im in images]
#         if masks is not None:
#             scaled_masks = [np.array(Image.fromarray(mask.astype(np.uint8)).resize((scaled_w, scaled_h))).astype(np.float32) for mask in masks]
#         else:
#             scaled_masks = None

#         offset_y = np.random.randint(scaled_h - in_h + 1)
#         offset_x = np.random.randint(scaled_w - in_w + 1)
#         cropped_images = [im[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for im in scaled_images]
#         if masks is not None:
#             cropped_masks = [mask[offset_y:offset_y + in_h, offset_x:offset_x + in_w] for mask in scaled_masks] 
#         else:
#             cropped_masks = None
        
#         output_intrinsics[0, 2] -= offset_x
#         output_intrinsics[1, 2] -= offset_y

#         # print(f"scale crop {cropped_masks[0].shape}")
#         return cropped_images, output_intrinsics, cropped_masks
