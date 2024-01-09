import cv2
import os 

def show_image_sequence(image_folder):
    image_list = sorted(os.listdir(image_folder))

    for image in image_list:
        # print(image)
        img = cv2.imread(os.path.join(image_folder, image))
        # print(img.shape)
        cv2.imshow('img',img)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    
    cv2.destroyAllWindows()

if __name__=='__main__':
    image_folder = '/home/ayon/thesis/data/Eiffel-Tower/2018/images'
    show_image_sequence(image_folder)