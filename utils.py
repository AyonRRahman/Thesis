import cv2
import pandas as pd 
import utm

import os 

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



def convert_latlon_utm(dataset_folder: str):
    '''
    this function takes a folder containing navigation.txt file and converts the lat,lon
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

    #save to csv
    df.to_csv(os.path.join(dataset_folder,'navigation.csv'))



if __name__=='__main__':
    dataset_folder = '/home/ayon/thesis/data/Eiffel-Tower/'
    for year in os.listdir(dataset_folder):
        print(year)
        dataset = os.path.join(dataset_folder,year)
        convert_latlon_utm(dataset)
        

    