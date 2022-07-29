import torch
import pandas as pd
import os
from PIL import Image
import torchvision
from torch.utils.data import Dataset, DataLoader

class only_nsv_360_images_dataset(Dataset):
    def __init__(self, transform = None):
        self.path_list = []
        for filename in os.listdir('../NSV_Processed_Data/Processed_360_JPEG_Images/Mahulpal-Jiral-L1/Romdas360/'):
            if filename[ - 4 : len(filename)] == '.jpg':
                self.path_list.append(f'../NSV_Processed_Data/Processed_360_JPEG_Images/Mahulpal-Jiral-L1/Romdas360/{filename}')
        for filename in os.listdir('../NSV_Processed_Data/Processed_360_JPEG_Images/Mahulpal-Jiral-R1/Romdas360/'):
            if filename[ - 4 : len(filename)] == '.jpg':
                self.path_list.append(f'../NSV_Processed_Data/Processed_360_JPEG_Images/Mahulpal-Jiral-R1/Romdas360/{filename}')
        
    def __len__(self):
        # return 1
        return len(self.path_list)

    def __getitem__(self, idx):
        im = Image.open(self.path_list[idx])
        # im = Image.open('../NSV_Processed_Data/Processed_360_JPEG_Images/Mahulpal-Jiral-R1/Romdas360/Mahulpal-Jiral-R1-Romdas360-0-00001.jpg')
        width, height = im.size
        im = im.crop((0, 0.1*height, width, height))
        pil_to_tensor = torchvision.transforms.ToTensor()
        im = pil_to_tensor(im)
        return im

def get_only_nsv_360_images_dataloaders(batch_size, shuffle_data = True):
    dataset = only_nsv_360_images_dataset()
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [int(0.9*dataset.__len__()), dataset.__len__() - int(0.9*dataset.__len__())])
    train_dataloader = DataLoader(train_dataset, shuffle = shuffle_data, num_workers = 4, batch_size = batch_size, drop_last=True)
    test_dataloader = DataLoader(test_dataset, shuffle = shuffle_data, num_workers = 4, batch_size = batch_size, drop_last=True)
    return train_dataloader, test_dataloader

# get_only_nsv_360_images_dataloaders(4, True)
# dataset = only_nsv_360_images_dataset()
# dataset.__getitem__(88)
