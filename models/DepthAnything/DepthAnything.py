from dpt import DepthAnything

from path import Path 
from tqdm import tqdm

import torch
import torch.nn as nn
from torchvision.transforms import Compose
import torchvision.transforms as tv_transform
import numpy as np


import matplotlib.pyplot as plt
from imageio.v2 import imread, imsave
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_tensor_image(filename):
    
    global DEVICE
    img = imread(filename).astype(np.float32)
    
    h, w, _ = img.shape
    
    img = np.transpose(img, (2, 0, 1))
    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.5)/0.5).to(DEVICE)
    return tensor_img

class Resize(object):
    
    """
    Adapted from https://github.com/LiheYoung/Depth-Anything/blob/main/depth_anything/util/transform.py
        Resize sample to given size (width, height).
    """
    def __init__(
        self,
        width=518,
        height=518,
        img_width = 464,
        img_height = 256,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
    ):
        """Init.

        Args:
            width (int): desired output width
            height (int): desired output height

            keep_aspect_ratio (bool, optional):
                True: Keep the aspect ratio of the input sample.
                Output sample might not have the given width and height, and
                resize behaviour depends on the parameter 'resize_method'.
                Defaults to True.

            ensure_multiple_of (int, optional):
                Output width and height is constrained to be multiple of this parameter.
                Defaults to 1.
            resize_method (str, optional):
                "lower_bound": Output will be at least as large as the given size.
                "upper_bound": Output will be at max as large as the given size. (Output size might be smaller than given size.)
                "minimal": Scale as least as possible.  (Output size might be smaller than given size.)
                Defaults to "lower_bound".
        """
        self.__width = width
        self.__height = height


        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method

        
        self.width, self.height = self.get_size(img_width, img_height)
        self._resize = tv_transform.Resize((int(self.height), int(self.width)))

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        # determine new height and width
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                # scale such that output size is lower bound
                if scale_width > scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                # scale such that output size is upper bound
                if scale_width < scale_height:
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                # scale as least as possbile
                if abs(1 - scale_width) < abs(1 - scale_height):
                    # fit width
                    scale_height = scale_width
                else:
                    # fit height
                    scale_width = scale_height
            else:
                raise ValueError(
                    f"resize_method {self.__resize_method} not implemented"
                )

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, min_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, min_val=self.__width
            )
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(
                scale_height * height, max_val=self.__height
            )
            new_width = self.constrain_to_multiple_of(
                scale_width * width, max_val=self.__width
            )
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def __call__(self, sample):
        sample = self._resize(sample)        
        return sample
    
class DepthAnythingSFM(nn.Module):
    def __init__(self, encoder='vitl', size=(256, 464)):
        assert encoder in ['vits', 'vitb', 'vitl']
        super(DepthAnythingSFM, self).__init__()
        self.model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).train()
        self.total_params = sum(param.numel() for param in self.model.parameters())
        print('Total Depth Anything parameters: {:.2f}M'.format(self.total_params / 1e6)) 
        self.transform = Compose([Resize()])
        self._size = size

    def forward(self, x):
        x = self.transform(x)
        depth = self.model(x)
        depth = torch.tensor(255) - depth
        depth = tv_transform.functional.resize(depth, self._size)

        return depth
    

if __name__=='__main__':
    dir = Path('/mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_Downscaled_colmap/2015')
    filenames = dir.glob("*.png")
    print(len(filenames))

    model = DepthAnythingSFM('vits')
    image = load_tensor_image(filenames[0])
    print(f"image shape {image.shape}")
    depth = model(image)
    print(depth.shape)
    depth = depth.detach().cpu().numpy().squeeze()
    depth = 255*(depth/depth.max())
    plt.imshow(255-depth)
    plt.colorbar()
    plt.show()



