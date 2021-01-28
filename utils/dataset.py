import os
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils
import colour
from colour.plotting import *

import rawpy    

from colour_demosaicing import demosaicing_CFA_Bayer_bilinear

class Raw_Dataset(Dataset):
    
    """Raw dataset with two classes labeled as 1 and 0"""

    def __init__(self, csv_file, root_dir, class_name, transform=None):
        
        """
        Args:
            csv_file (string): Path to the csv file with file name and label.
            root_dir (string): Directory with all the images.
            class_name (string): name of the class with label 1
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.class_name = class_name
        self.transform = transform

        
    def __len__(self):
        return len(self.df)

    
    def __getitem__(self, idx):
        
        """
        Args:
            idx (int): refers to row in csv_file of item
        return:
            tuple of:
                image (rawpy._rawpy.RawPy)
                label (int) with value in {0,1}
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image_name = os.path.join(self.root_dir, self.df.iloc[idx]['file name'] + '.npy')
        
        image_raw_visible =  np.load(image_name)
        
        if self.transform:
            im = self.transform(image_raw_visible)
        
        label = 0
        if self.class_name == self.df['label'][idx]:
            label = 1
         
        sample = (im,label)
            
        return sample
    
class ToTensor(object):
    
    """demosaic rawpy._rawpy.RawPy image with a debayer algorithm from colour_demosaicing"""
    
    def __init__(self, dtype=np.int16, stacked=False):
        
        """
        Args:
            demosaic (function): debayer algorithm from colour_demosaicing
            pattern (string): Arrangement of the colour filters on the pixel array
        """

        self.dtype = dtype
        self.stacked = stacked

    def __call__(self, im):
        
        """
        Args:
            im (rawpy._rawpy.RawPy)
        return:
            torch.tensor from output image of the demosaic algo transposed from (H,W,3) to (3,H,W)
        """
        
        im.dtype = self.dtype
        if self.stacked:
            im = np.squeeze(im)
            im = np.stack((im,im,im),axis=0)  
        
        return torch.tensor(im,dtype=torch.float)
    
class Demosaicing(object):
    
    """demosaic rawpy._rawpy.RawPy image with a debayer algorithm from colour_demosaicing"""
    
    def __init__(self, demosaic=demosaicing_CFA_Bayer_bilinear, pattern='RGGB'):
        
        """
        Args:
            demosaic (function): debayer algorithm from colour_demosaicing
            pattern (string): Arrangement of the colour filters on the pixel array
        """

        self.demosaic = demosaic
        self.pattern = pattern

    def __call__(self, im):
        
        """
        Args:
            im (rawpy._rawpy.RawPy)
        return:
            torch.tensor from output image of the demosaic algo transposed from (H,W,3) to (3,H,W)
        """
        
        mosaic = np.squeeze(im)
        demosaic = self.demosaic(mosaic,pattern=self.pattern) #image_color is of size (H,W,3)
        image = demosaic.transpose(2,0,1) 
        
        return torch.tensor(image,dtype=torch.float)
    
class RAISE2_Dataset(Dataset):
    
    """RAISE2 dataset with two classes labeled as 1 and 0"""

    def __init__(self, csv_file, root_dir, class_name, transform=None):
        
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            class_name (string): name of the class with label 1
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        
        self.RAISE2 = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.class_name = class_name
        self.transform = transform

        
    def __len__(self):
        return len(self.RAISE2)

    def __getitem__(self, idx):
        
        """
        Args:
            idx (int): refers to row in csv_file of item
        return:
            tuple of:
                image (rawpy._rawpy.RawPy)
                label (int) with value in {0,1}
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.root_dir, self.RAISE2.iloc[idx]['File'] + '.NEF')

        im = rawpy.imread(img_name)
        
        if self.transform:
            im = self.transform(im)
        
        label = 0
        if self.class_name in self.RAISE2['Keywords'][idx]:
            label = 1
            
        sample = (im,label)
            
        return sample
    
class Demosaicing_Raise(object):
    
    """demosaic rawpy._rawpy.RawPy image with a debayer algorithm from colour_demosaicing"""
    
    def __init__(self, demosaic=demosaicing_CFA_Bayer_bilinear, pattern='RGGB'):
        
        """
        Args:
            demosaic (function): debayer algorithm from colour_demosaicing
            pattern (string): Arrangement of the colour filters on the pixel array
        """

        self.demosaic = demosaic
        self.pattern = pattern

    def __call__(self, im):
        
        """
        Args:
            im (rawpy._rawpy.RawPy)
        return:
            torch.tensor from output image of the demosaic algo transposed from (H,W,3) to (3,H,W)
        """
        
        mosaic = im.raw_image
        
        image_color = self.demosaic(mosaic,pattern=self.pattern) #image_color is of size (H,W,3)
        
        image  = image_color.transpose((2, 0, 1)) #image is of size (3,H,W)

        return torch.tensor(image,dtype=torch.float)

def get_dataset_stats(dataset,channel=3):  
    
    """
    Calculate the mean and the std for each channel in the dataset
    """
    
    mean = torch.zeros(channel)
    squared_mean =torch.zeros(channel)
    
    N = len(dataset)
    
    for i in range(N):
        X = dataset[i][0]
        mean += (1/N)*torch.mean(X,axis=(1,2))
        squared_mean += (1/N)*torch.mean(X**2,axis=(1,2))
        
    std = torch.sqrt(squared_mean - mean**2)
  
    return mean, std

def imshow(image):
    
    image = image.numpy()
    image = image.transpose(1,2,0)
    plot_image(image/np.max(image))