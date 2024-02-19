import sys
import os 

#setting the path to the project root
script_path = os.path.abspath(__file__)
project_folder = os.path.abspath(os.path.join(os.path.dirname(script_path), '..'))
sys.path[0] = project_folder

# print(sys.path)
import numpy as np
import torch
import shutil
import cv2
from tqdm import tqdm
from path import Path 
from PIL import Image, ImageOps
from imageio import imread

from utils.colmap_script import read_cameras_binary
from datasets.image_loader import ImageLoader


def get_cam_txt(dir: str) -> np.ndarray:
    '''
    This function converts the cameras.bin file generated by colmap undistort image command
    and creates the intrinsic camera matrix and saves it as cam.txt file.

    params:
        dir: path to the folder where colmap saved after undistorting images.
            should contain images, sparse etc. folder
    
    Returns:
        intrinsic_matrix: 3x3 np array representing camera intrinsic matrix
    
    '''
    try:
        assert  os.path.exists(os.path.join(dir, 'sparse/cameras.bin'))
    except:
        print(f"{os.path.join(dir, 'sparse/cameras.bin')} does not exist.")
        raise ValueError(f"Folder {os.path.join(dir, 'sparse/cameras.bin')} does not exist.")


    cameras = read_cameras_binary(os.path.join(dir, 'sparse/cameras.bin'))
    for x in cameras:
        assert cameras[x].model=='PINHOLE'
        assert len(cameras[x].params)==4

        fx = cameras[x].params[0]
        fy = cameras[x].params[1]
        cx = cameras[x].params[2]
        cy = cameras[x].params[3]

        intrinsic_matrix = np.array([[fx, 0, cx],[0, fy, cy], [0, 0, 1]])
        return intrinsic_matrix
        # np.savetxt(os.path.join(dir, 'images/cam.txt'), intrinsic_matrix)


@torch.no_grad() 
def get_image_stats(dataset, batch_size:int = 10, stat_all:bool=False):
    '''
    get statistics of the dataset for normalizing.
    
    params:
        dataset (pytorch dataset class): a pytroch dataset class instance that returns images
        batch_size (int): batch size for the dataloader
    '''
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

    # placeholders
    ch_1_sum = 0
    ch_2_sum = 0
    ch_3_sum = 0

    ch_1_sq_sum = 0
    ch_2_sq_sum = 0
    ch_3_sq_sum = 0

    #calculating mean and std for each channel separately

    # loop through images
    _,w,h = dataset[0].shape

    for data in loader:
        for images in data[0]:
            ch_1_sum += images[0].sum()
            ch_2_sum += images[1].sum()
            ch_3_sum += images[2].sum()

        # pixel count
        count = batch_size*w*h
        # mean and std
        ch1_mean = ch_1_sum/ count
        ch2_mean = ch_2_sum/ count
        ch3_mean = ch_3_sum/ count
        
        for images in data[0]:
            ch_1_sq_sum += ((images[0]-ch1_mean)**2).sum()
            ch_2_sq_sum += ((images[1]-ch1_mean)**2).sum()
            ch_3_sq_sum += ((images[2]-ch1_mean)**2).sum()

        
        if not stat_all:
            break


    #total pixel count
    if stat_all:
        count = len(dataset)*w*h
    else:
        count = batch_size*w*h

    # mean and std

    ch1_var  = (ch_1_sq_sum/ count)
    ch1_std  = torch.sqrt(ch1_var)

    ch2_var  = (ch_2_sq_sum/ count)
    ch2_std  = torch.sqrt(ch2_var)

    ch3_var  = (ch_3_sq_sum/ count)
    ch3_std  = torch.sqrt(ch3_var)

    # output
    print('mean: '  + str([ch1_mean.item(), ch2_mean.item(), ch3_mean.item()]))
    print('std:  '  + str([ch1_std.item(), ch2_std.item(), ch3_std.item()]))
    
    return {'mean': [ch1_mean.item(), ch2_mean.item(), ch3_mean.item()], 'std':[ch1_std.item(), ch2_std.item(), ch3_std.item()]}

def Downscale_image(root='./data/Eiffel-Tower_ready_opencv',out='./data/Eiffel-Tower_ready_Downscaled', scale=0.25):
    
    '''
    Downscale the images and transforms the cam.txt accordingly
    in root folder by given scale and saves it in the out folder.

    Params:
        root(str):      root directory of the data
        out(str):       output directory to save the data
        scale(float):   scale to be used for downscaling
    '''
    if not os.path.isdir(root):
        raise ValueError(f'{root} is not a valid directory')
    root = Path(root)
    out = Path(out)
    out.makedirs_p()    
    folders = os.listdir(root)
    scaling_matrix = np.array([[scale, 0, 0],[0, scale, 0], [0 , 0, 1]])
    print(f'scaling matrix for camera intrinsic:\n {scaling_matrix}')
    for folder in folders:
        # print(folder)
        if not folder=='2015':
            print(folder)
            continue
        
        #copy the val.txt and train.txt
        if not os.path.isdir(root/folder):
            shutil.copy2(root/folder, out)
            continue
        
        (out/folder).makedirs_p()
        #open the images in the folder and downscale them and save them in the destination
        for images in tqdm(sorted(os.listdir(root/folder))):
            #change the cam.txt
            if images=='cam.txt':
                K = np.loadtxt(root/folder/images)
                new_K = scaling_matrix@K
                np.savetxt(out/folder/'cam.txt', new_K)
                continue
            #other wise rescale image and save
            img = cv2.imread(root/folder/images)
            downscaled_img = cv2.resize(img, dsize=(int(scale*img.shape[1]), int(scale*img.shape[0])), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(out/folder/images, downscaled_img) 
            
def center_crop(img, dim):
    """Returns center cropped image
    source:
    https://medium.com/curious-manava/center-crop-and-scaling-in-opencv-using-python-279c1bb77c74

    Args:
    img: image to be center cropped
    dim: dimensions (width, height) to be cropped from center
    """
    width, height = img.shape[1], img.shape[0]

	# process crop width and height for max available dimension

    crop_width = dim[0] if dim[0]<img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1]<img.shape[0] else img.shape[0] 
    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2) 
    crop_img = img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]
    return crop_img


def crop_and_downscale_colmap_undistorted_image(root='data/Eiffel-Tower_undistorted', out='data/Eiffel-Tower_ready_Downscaled_colmap/', scale=0.25):
    '''
    Downscale after center cropping of the the images and 
    transforms the cam.txt accordingly in root folder by 
    given scale and saves it in the out folder.

    Params:
        root(str):      root directory of the data
        out(str):       output directory to save the data
        scale(float):   scale to be used for downscaling <1
    '''
    
    
    root = Path(root)
    folders = os.listdir(root)
    required_shape = (64*29, 64*16)  #(wd, ht)
    #for all the folders found with 
    # # print(img.shape[0]//64)
    # # print(img.shape[1]//64)
    output_folder = Path(out)
    output_folder.makedirs_p()
    depth_output_directory = Path('./data/scaled_and_cropped_depth/')
    depth_output_directory.makedirs_p()

    #saving train.txt and val.txt
    with open(output_folder/'train.txt', 'w') as file:
        file.write("2018\n")
        file.write("2016\n")
        file.write("2020\n")

    with open(output_folder/'val.txt', 'w') as file:
        file.write("2015\n")

    for folder in folders:
        img_directory = root/folder/'images'
        imgs = img_directory.glob('*.png')
        print(f"found {len(imgs)} images in {folder} folder")
        img = cv2.imread(imgs[0])
        
        print(f"shape of the images {img.shape}")

        ht_to_crop = img.shape[0]-required_shape[1]
        wd_to_crop = img.shape[1]-required_shape[0]

        
        # print(cropped_image.shape)
        cam_intrinsic = get_cam_txt(root/folder)
        
        # modify cam_intrinsic for center cropping
        cam_intrinsic[0,2] = cam_intrinsic[0,2]-wd_to_crop/2 
        cam_intrinsic[1,2] = cam_intrinsic[1,2]-ht_to_crop/2
        
        # print(cam_intrinsic)
        # modify the cam intrinsic for scaling
        scaling_matrix = np.array([[scale, 0, 0],[0, scale, 0], [0 , 0, 1]])
        CroppedAndScaled_cam_intrinsic = scaling_matrix@cam_intrinsic
        #make the folder
        (output_folder/folder).makedirs_p()
        #save the cam.txt file
        np.savetxt(output_folder/folder/'cam.txt', CroppedAndScaled_cam_intrinsic)
        for image in tqdm(sorted(imgs)):
            #load image
            img = cv2.imread(image)
            cropped_image = center_crop(img, required_shape) 
            scaled_img = cv2.resize(cropped_image, dsize=(int(scale*cropped_image.shape[1]), int(scale*cropped_image.shape[0])), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite(output_folder/folder/image.split('/')[-1], scaled_img) 
            
        print(f"output image shape {scaled_img.shape}")
        depth_images = (root/folder/'depth_images').glob('*.png')
        depth_folder = depth_output_directory/folder
        depth_folder.makedirs_p()
        
        for depths in tqdm(sorted(depth_images)):  
            img = cv2.imread(depths).astype(np.uint8)
            cropped_image = center_crop(img, required_shape) 
            scaled_img = cv2.resize(cropped_image, dsize=(int(scale*cropped_image.shape[1]), int(scale*cropped_image.shape[0])), interpolation=cv2.INTER_NEAREST)
            
            cv2.imwrite(depth_folder/depths.split('/')[-1], scaled_img) 
            

        print(f"output depth shape {scaled_img.shape}")
        
            

        
        


        



        
if __name__=="__main__":
    
    print('!!!!!!!')
    crop_and_downscale_colmap_undistorted_image()
    # Downscale_image()
    