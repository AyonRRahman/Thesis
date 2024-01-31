import torch
import torch.utils.data as data
import os
from path import Path
from PIL import Image
import torchvision.transforms as T

class ImageLoader(data.Dataset):
    '''
    A dataset class to load all the images in all the folders in root folder
    '''
    
    def __init__(self, root='/mundus/mrahman527/Thesis/data/Eiffel-Tower_ready_opencv'):
        self.root = Path(root)
        self.crawl_folders()
        self.transform = T.Compose([T.ToTensor()])

    def crawl_folders(self):
        images = []
        folders = os.listdir(self.root)
        for x in folders:
            if os.path.isdir(self.root/x):
                folder_path = self.root/x
                image_list = os.listdir(folder_path)
                # print(sorted(image_list))
                image_list.remove('cam.txt')
                # print(sorted(image_list))

                for image in image_list:
                    images.append(folder_path/image)

        self.samples = images
        

    def __getitem__(self, index):
        sample = self.samples[index]
        img =  Image.open(sample).convert('RGB')

        return self.transform(img)
        
        
        

    def __len__(self):
        return len(self.samples)


if __name__=='__main__':
    dataset = ImageLoader()
    print(dataset[0].shape)


    