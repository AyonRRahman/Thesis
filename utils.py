import cv2
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

if __name__=='__main__':
    image_folder = '/home/ayon/thesis/data/Eiffel-Tower/2018/images'
    show_image_sequence(image_folder)