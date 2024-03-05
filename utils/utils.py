import cv2
import pandas as pd 
import numpy as np
import utm
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Rotation as Rot

from tqdm import tqdm
import os 
from path import Path
import imageio
from PIL import Image

def show_image_with_depth(img_folder, depth_folder, resize=True, save_video=False):
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
    frames = []
    print(f'{len(images)}, {len(depths)}')
    for i,(img, depth) in tqdm(enumerate(zip(images, depths))):
        # print(img)
        # print(os.path.isfile(img))
        img_loaded = cv2.imread(img)
        if save_video:
            img_loaded = cv2.cvtColor(img_loaded, cv2.COLOR_BGR2RGB)
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
        # print(depth_norm.shape)
        # print(img_loaded.shape)
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
            img_to_show = np.hstack((img_loaded/255.0, depth_padded))
            # print(f"img show shape{img_to_show.shape}")
            
            
        
        else:
            img_to_show = np.hstack((img_loaded/255.0, depth_norm))
            # print(f"img show shape{img_to_show.shape}, depth_norm {depth_norm.shape}, img {img_loaded.shape}")

        
        if save_video:
            # print(img_to_show.shape)
            frames.append(Image.fromarray((img_to_show * 255).astype(np.uint8)))
        # print(f'dep {depth_norm.shape}')
        if not save_video:
            cv2.imshow('img and gt_depth', img_to_show)
        # print(img)
        # cv2.waitKey(1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            break
    
    cv2.destroyAllWindows()

    if save_video:
        print("Saving GIF file")
        with imageio.get_writer("depth_prediction.gif", mode="I") as writer:
            for idx, frame in enumerate(frames):
                print("Adding frame to GIF file: ", idx + 1)
                writer.append_data(frame)



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


def export_trajectory(file, output_dir = 'data/Eiffel-Tower/2015/'):
    '''
    Takes sfm/images.txt file and exports the gt pose in a gt_traj.txt file in the output_dir folder.

    exports the 4x4 pose matrix as flattend
    Every row of the file contains the first 3 rows of a 
    4x4 homogeneous pose matrix (SE(3) matrix) flattened 
    into one line, with each value separated by a space. 
    (Kitty format)
    
    For example, this pose matrix: 
    a b c d
    e f g h
    i j k l
    0 0 0 1
    >>>>>>> 
    a b c d e f g h i j k l
    '''
    assert os.path.exists(file)
    gt_df = Eiffel_save_navigation_data(file).sort_index()
    # print(gt_df.head())
    poses = np.zeros((1,12))
    # print(poses)

    for index, row in gt_df.iterrows():
        x = row['TX']
        y = row['TY']
        z = row['TZ']

        trans = np.array((x,y,z)).reshape(3,1)
        
        rx = row['Rx']
        ry = row['Ry']
        rz = row['Rz']
        
        rot_z = R.from_euler('z', rz, degrees=True).as_matrix()
        rot_x = R.from_euler('x', rx, degrees=True).as_matrix()
        rot_y = R.from_euler('y', ry, degrees=True).as_matrix()
        rot_mat = rot_z@rot_y@rot_x

        inverse_pose_translation = -(rot_mat.T)@trans
        inverse_pose_rot = rot_mat.T
        
        inverse_pose = np.concatenate((inverse_pose_rot, inverse_pose_translation), axis=1)
        poses = np.concatenate((poses, np.expand_dims(inverse_pose.flatten(), axis=0)), axis=0)
        
    np.savetxt(os.path.join(output_dir,'gt_traj.txt'),poses[1:],fmt="%f", delimiter=' ')



def compute_Twc_from_Tcw(tx,ty,tz,qx,qy,qz,qw):
    tcw = np.array([[tx,ty,tz]]).reshape(3,1)
    qcw = np.array([[qx,qy,qz,qw]]).reshape(4,)
    rot_cw = Rot.from_quat(qcw)
    Rcw = rot_cw.as_matrix()
    Rwc = Rcw.T
    twc = -Rwc.dot(tcw)
    rot_wc = Rot.from_matrix(Rwc)
    qwc = rot_wc.as_quat()
    return twc, qwc

def extract_cam_pose_from_line(line):
    line_els = line.split(" ")
    img_id = int(line_els[0])
    qw = float(line_els[1])
    qx = float(line_els[2])
    qy = float(line_els[3])
    qz = float(line_els[4])
    tx = float(line_els[5])
    ty = float(line_els[6])
    tz = float(line_els[7])
    cam_id = int(line_els[8])
    # print(line_els[9].split(".")[0].split("_")[-1])
    img_name = line_els[9].split(".")[0].split("_")[-1]
    twc, qwc = compute_Twc_from_Tcw(tx,ty,tz,qx,qy,qz,qw)
    tx = twc[0,0]
    ty = twc[1,0]
    tz = twc[2,0]
    qx = qwc[0]
    qy = qwc[1]
    qz = qwc[2]
    qw = qwc[3]
    return img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name


def get_trajectory(file):
    tx_list = []
    ty_list = []
    tz_list = []
    name_list = []

    with open(file) as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line[0]=='#':
                continue
            
            if i%2==0:
                # print(line)
                # print(len(line.split(' ')))
                # print(extract_cam_pose_from_line(line))
                img_id, qw, qx, qy, qz, tx, ty, tz, cam_id, img_name = extract_cam_pose_from_line(line)
                tx_list.append(tx)
                ty_list.append(ty)
                tz_list.append(tz)
                name_list.append(img_name)

        df = pd.DataFrame()
        df['image']=name_list
        df['x']=tx_list
        df['y']=ty_list
        df['z']=tz_list
        df = df.sort_values('image')

        return (df['x'].to_list(), df['y'].to_list(), df['z'].to_list())



if __name__=='__main__':
    # image_path = Path('data/Eiffel-Tower_ready_Downscaled/2015')
    # depth_path = Path('data/Eiffel-Tower_depth_images/2015/depth_images')
    # # image_path = Path('depth_evaluation/equal_wrights_b16_sl3_lr1e-4/2015')
    # show_image_with_depth(image_path, depth_path, resize=True, save_video=False)
    # export_trajectory(file='/mundus/mrahman527/Thesis/data/Eiffel-Tower/2018/sfm/images.txt', output_dir = '/mundus/mrahman527/Thesis/data/Eiffel_tower_ready_small_set/2018')
    pass