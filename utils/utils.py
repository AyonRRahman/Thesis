import cv2
import pandas as pd 
import numpy as np
import utm
from scipy.spatial.transform import Rotation as R

from tqdm import tqdm
import os 
from path import Path


def show_image_with_depth(img_folder, depth_folder, resize=True):
    '''
    function to show images and depth side by side
    args:
        img_folder(str): path to image folder
        depth_folder(str): path to depth folder
        resize(bool): should the function resize depth 
    '''
    image_path = Path(img_folder)
    images = sorted(image_path.glob('*.png'))
    depth_path = Path(depth_folder)
    depths = sorted(depth_path.glob('*.png'))

    for i,(img, depth) in tqdm(enumerate(zip(images, depths))):
        # print(img)
        # print(os.path.isfile(img))
        img_loaded = cv2.imread(img)
        # img_list.append(img_loaded)
        # print(type(img_loaded))
        # print(img_loaded.shape)
        # cv2.imshow('img',img_loaded)
        # print(f'img {img_loaded.shape}')

        depth_loaded = cv2.imread(depth)
        # print(type(depth_loaded))
        if resize:
            depth_loaded = cv2.resize(depth_loaded, (depth_loaded.shape[1]//4,depth_loaded.shape[0]//4) )

        depth_norm = depth_loaded/depth_loaded.max()
        
        #check which size is bigger
        diff = depth_norm.shape[0] - img_loaded.shape[0]
        if diff>0: #depth is bigger
            zeros_to_pad = np.zeros((diff, img_loaded.shape[1],3), dtype=np.uint8)
            img_padded = np.vstack((img_loaded, zeros_to_pad))
            # print(f'err {np.sum(img_loaded - img_padded[:270,:,:])}')
            # cv2.imshow('img',img_padded)
            # print(depth_norm.dtype)
            # print(img_loaded.dtype)
            # print(f'img_padded {img_padded.shape}')
            img_to_show = np.hstack((img_padded/255.0, depth_norm))
        elif diff<0:
            zeros_to_pad = np.zeros((abs(diff), depth_norm.shape[1],3), dtype=np.uint8)
            depth_padded = np.vstack((depth_norm, zeros_to_pad))
            img_to_show = np.vstack((img_loaded/255.0, depth_padded))
        
        else:
            img_to_show = np.vstack((img_loaded/255.0, depth_norm))

        
        # print(f'dep {depth_norm.shape}')
        cv2.imshow('img and gt_depth', img_to_show)
        # print(img)
        # cv2.waitKey(1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break

    cv2.destroyAllWindows()



def show_image_sequence(image_folder: str):
    '''
    This function takes a folder containing image sequence and visualize them. 
    Args:
        image_folder (str): Path to the folder containing image sequence.

    '''
    if not isinstance(image_folder, str):
        raise ValueError("Please provide a string as the image_folder argument.")
    
    if not os.path.exists(image_folder):
        print(f"The folder {image_folder} does not exist.")
        return 
    
    image_list = sorted(os.listdir(image_folder))

    for image in image_list:        
        img = cv2.imread(os.path.join(image_folder, image))
        cv2.imshow('img',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Terminated by user')
            cv2.destroyAllWindows()
            break
    
    cv2.destroyAllWindows()



def Eiffel_convert_latlon_utm(dataset_folder: str):
    '''
    For Eiffel Tower dataset.
    This function takes a folder containing navigation.txt file and converts the lat,lon
    system to utm projection system and saves the result as a csv file. 

    Args:
        dataset_folder (str): Path to the folder containing navigation.txt file.

    '''
    if not isinstance(dataset_folder, str):
        raise ValueError("Please provide a string as the image_folder argument.")
    
    
    if not os.path.exists(dataset_folder):
        print(f"The folder {dataset_folder} does not exist.")
        return 
    
    if not os.path.exists(os.path.join(dataset_folder, 'navigation.txt')):
        print(f"There is no navigation.txt file in the folder {dataset_folder}.")
        return 
    

    #read the navigation.txt
    df = pd.read_csv(os.path.join(dataset_folder, 'navigation.txt'), delimiter=' ',index_col='name')

    east = []
    north = []

    #convert latlon to utm 
    for index, rows in df.iterrows():
        lat = rows['lat']
        lon = rows['lon']
        easting, northing, zone_number,_ = utm.from_latlon(lat, lon)
        east.append(easting)
        north.append(northing)
    
    df['north'] = north
    df['east'] = east

    depth = df['alt'].to_numpy()

    # calculate the relative position w.r.t to the initial position 
    df['north_relative'] = np.array(north) - north[0]
    df['east_relative'] = np.array(east) - east[0]
    df['depth_relative'] = depth - depth[0]

    #save to csv
    return df
    # df.to_csv(os.path.join(dataset_folder,'navigation.csv'))


def Eiffel_save_navigation_data(file_path: str) -> pd.DataFrame:

    '''
    This function takes images.txt file from the Eiffel tower dataset and saves a csv file
    containing the trajectory along wih euler angles and quaternions. 
    Args:
        file_path (str): Path to the images.txt file.

    Returns:
        df (pd.DataFrame): Processed dataframe

    '''

    if os.path.exists(file_path):

        #list to store the data
        data = []

        #open and process the txt file
        with open(file_path, 'r') as f:
            for index, line in enumerate(f):
                if index<4:
                    continue
            
                if index%2!=0:
                    continue
                
                line_content = line.split(' ')
                
                image_name = line_content[-1]
                QW = line_content[1]
                QX = line_content[2]
                QY = line_content[3]
                QZ = line_content[4]
                TX = line_content[5]
                TY = line_content[6]
                TZ = line_content[7]
                Camera_id = line_content[8]


                # get euler angles from quaternions
                r = R.from_quat([QW, QX, QY, QZ])
                Rz, Ry, Rx = r.as_euler('zyx', degrees=True)
                
                #append the data in the list
                data.append([image_name[:-1], QW, QX, QY, QZ, TX, TY, TZ, Rx, Ry, Rz])

        # make the dataframe and process it for saving
        df = pd.DataFrame(data, columns=['image_name', 'QW', 'QX', 'QY', 'QZ', 'TX', 'TY', 'TZ', 'Rx', 'Ry', 'Rz'])
        df = df.sort_values(by='image_name')
        df = df.set_index('image_name')
        df = df.astype(float)
        return df
    
    else:
        print(f"The file '{file_path}' does not exist.")
        return None


if __name__=='__main__':
    image_path = Path('data/Eiffel-Tower_ready_Downscaled/2015')
    depth_path = Path('data/Eiffel-Tower_depth_images/2015/depth_images')

    show_image_with_depth(image_path, depth_path)
