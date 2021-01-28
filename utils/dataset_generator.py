"""
Prepare a dataset for the neural network
"""
__author__ = "Marco Aversa"

from skimage.util.shape import view_as_windows
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

def np2torch(array):
    """Return a numpy array in a torch tensor"""    
    return torch.from_numpy(array).float()

def split_img(imgs, ROIs = (3,3) , step= (1,1)):
    """Split the imgs in regions of size ROIs.

       Args:
            imgs (ndarray/tensor): images which you want to split --> img size is: (Batch,Channels,Height,Weight)
            ROIs (tuple): size of sub-regions splitted (ROIs=region of interests)
            step (tuple): step path from one sub-region to the next one (in the x,y axis)
       
        Returns:
            ndarray: splitted subimages. 
                     The size is (x_num_subROIs*y_num_subROIs, ROIs[0]*ROIs[1]) where:
                     x_num_subROIs = ( imgs.shape[1]-int(ROIs[1]/2)*2 )/step[1]
                     y_num_subROIs = ( imgs.shape[0]-int(ROIs[0]/2)*2 )/step[0]                     
       
       Example:
            >>> from dataset_generator import split
            >>> imgs_splitted = split(imgs, ROI_size = (5,5), step=(2,3))
            
       Note: if the input is a tensor, it could be splitted only in square matrices
    """
    channels = imgs.shape[1]
    if type(imgs) == type(np.array(1)):
        if ROIs[0] != imgs.shape[2] or ROIs[1] != imgs.shape[3]:
            imgs = view_as_windows(imgs, (1, channels, ROIs[0],ROIs[1]), [1, channels, step[0],step[1]])
        splitted=imgs.reshape(-1, channels, imgs.shape[-2]*imgs.shape[-1])
    if type(imgs) == type(torch.tensor([1])):
        imgs=imgs.unfold(2,ROIs[0],step[0]).unfold(3,ROIs[0],step[0])
        splitted=imgs.reshape(-1,imgs.shape[4]*imgs.shape[5]).squeeze()
    return splitted

def join_blocks(splitted, final_shape, blocks_shape):
    """Join blocks to reobtain a splitted grey image
    
    Attribute:
        splitted (tensor) = splitted images in blocks. size = (Batch,Channels,Height,Weight)
        final_shape (tuple) = size of the final image reconstructed
        blocks_shape (tuple) = size of each block
    Return:
        tensor: image restored from blocks. size = (Height,Weight)
    
    """
    
    rows = int(final_shape[0]/blocks_shape[0])
    columns = int(final_shape[1]/blocks_shape[1])

    final_img = torch.empty(rows, 1, blocks_shape[0], blocks_shape[1]*columns)
    for r in np.arange(rows):    
        stackblocks = splitted[r*columns]
        for c in np.arange(1, columns):
            stackblocks = torch.cat((stackblocks, splitted[r*columns+c]), axis=2)
        final_img[r] = stackblocks
    return final_img.reshape(final_shape[0],final_shape[1])

def normalize(imgs, ROIs=(3,3), step=(1,1)): 
    """Split the imgs in regions of size ROIs. Each region is normalized for its mean and standard deviation.
       If the image is already splitted, it just normalize
    
       Args:
            imgs (ndarray): images which you want to normalize --> img size is: (Batch,Channels,Weight,Height)
            ROIs (tuple): size of sub-regions splitted (ROIs=region of interests)
            step (tuple): step path from one sub-region to the next one (in the x,y axis)
            
       Returns:
            mu (ndarray): mean value of each subimage.
                          The size is (x_num_subROIs*y_num_subROIs, 1) 
            sigma (ndarray): standard deviation of each subimage.
                             The size is (x_num_subROIs*y_num_subROIs, 1) 
            normalized (ndarray): normalized subimages. 
                                  The size is (x_num_subROIs*y_num_subROIs, ROIs[0]*ROIs[1]) 
            imgs_shape (tuple): initial imgs shape. The images initial shape is: (batch, channels, heights, weights) 
                                  
            where:
                x_num_subROIs = ( imgs.shape[1]-int(ROIs[1]/2)*2 )/step[1]
                y_num_subROIs = ( imgs.shape[0]-int(ROIs[0]/2)*2 )/step[0]    
            
       Example:
            >>> from dataset_generator import normalize_imgs
            >>> mu, sigma, imgs_normalized, imgs_shape = normalize(imgs, ROI_size = (5,5), step=(2,3))
    """    
    
    #Split imgs in subimages of size ROIs
    
    imgs = imgs.astype('float32')
    
    channels=1
    tot_imgs=1
    
    imgs_shape = imgs.shape
    
    if len(imgs_shape) == 4:
        imgs = split_img(imgs, ROIs, step)
        tot_imgs = imgs.shape[0]
        channels = imgs.shape[1]
    
    #Compute the mean for each subregion  
    mu = np.mean(imgs,-1)
    if channels == 1:
        mu = mu[:,np.newaxis]

    #Compute the standard for each subregion  
    sigma = np.std(imgs,-1)  
    if channels == 1:
        sigma = sigma[:,np.newaxis]

    #Set std=0 to 1 to avoid nan in the normalization
    ind_zero_test = np.where(np.any(sigma==0, axis=1))
    sigma[ind_zero_test]=1

    #Normalize data
    normalized = np.empty(imgs.shape)
    
    for i in np.arange(tot_imgs):
        for j in np.arange(channels):
            normalized[i,j] = (imgs[i,j] - mu[i,j])/sigma[i,j]

    return mu, sigma, normalized, imgs_shape

def loader_sequential_generator(X,Y, batch_size, ROIs = (64,64), step = (64,64)):
    """ Generate the dataloader to feed the network with splitted sequential regions
        
        Args:
            Y (ndarray): target of your dataset --> size: (Batch,Channels,Weight,Height)
            X (ndarray): input of your dataset --> size: (Batch,Channels,Weight,Height)
            datasize = number of random ROIs to generate
            ROIs (tuple): size of random region (ROIs=region of interests)
            batch_size = batch size of the dataloader
            
        Returns:
            Dataloader: format to train the net with a batch size
          
    """ 

    mu, sigma, X_train, X_train_shape = normalize(X, ROIs, step)
#     _, _, Y_train, _ = normalize(Y, ROIs, step)
    
    Y_train = (split_img(Y, ROIs, step) - mu)/sigma
   
    X_train = np2torch(X_train).reshape(-1, X_train_shape[1], ROIs[0], ROIs[1])
    Y_train = np2torch(Y_train).reshape(-1, X_train_shape[1], ROIs[0], ROIs[1])

    dataset = TensorDataset(X_train, Y_train)

    return X_train, Y_train, np2torch(mu), np2torch(sigma), DataLoader(dataset, batch_size)

def random_ROI(X, Y, ROIs = (512,512)):
    """ Return a random region for each input/target pair images of the dataset
        Args:
            Y (ndarray): target of your dataset --> size: (Batch,Channels,Weight,Height)
            X (ndarray): input of your dataset --> size: (Batch,Channels,Weight,Height)
            ROIs (tuple): size of random region (ROIs=region of interests)
           
        Returns:
            For each pair images (input/target) of the dataset, return respectively random ROIs
            Y_cut (ndarray): target of your dataset --> size: (Batch,Channels,ROIs[0],ROIs[1])
            X_cut (ndarray): input of your dataset --> size: (Batch,Channels,ROIs[0],ROIs[1])
            
        Example:
            >>> from dataset_generator import random_ROI
            >>> X,Y = random_ROI(X,Y, ROIs = (10,10))
    """    
    
    X_cut=np.empty((X.shape[0],X.shape[1], ROIs[0], ROIs[1]))
    Y_cut=np.empty((Y.shape[0],Y.shape[1], ROIs[0], ROIs[1]))
    
    for batch in np.arange(len(X)):
        x_size=int(random.random()*(Y.shape[2]-(ROIs[0]+1)))
        y_size=int(random.random()*(Y.shape[3]-(ROIs[1]+1)))
        X_cut[batch]=X[batch, :, x_size:x_size+ROIs[0],y_size:y_size+ROIs[1]]
        Y_cut[batch]=Y[batch, :, x_size:x_size+ROIs[0],y_size:y_size+ROIs[1]]
    return X_cut, Y_cut

def one2many_random_ROI(X, Y, datasize=1000, ROIs = (512,512)):
    """ Return a dataset of N subimages obtained from random regions of the same image
        Args:
            Y (ndarray): target of your dataset --> size: (1,Channels,Weight,Height)
            X (ndarray): input of your dataset --> size: (1,Channels,Weight,Height)
            datasize = number of random ROIs to generate
            ROIs (tuple): size of random region (ROIs=region of interests)
           
        Returns:
            Y_cut (ndarray): target of your dataset --> size: (Datasize,Channels,ROIs[0],ROIs[1])
            X_cut (ndarray): input of your dataset --> size: (Datasize,Channels,ROIs[0],ROIs[1])
    """   

    X_cut=np.empty((datasize,X.shape[1],ROIs[0],ROIs[1]))
    Y_cut=np.empty((datasize,Y.shape[1],ROIs[0],ROIs[1]))

    for i in np.arange(datasize):
        X_cut[i], Y_cut[i] = random_ROI(X, Y, ROIs)
    return X_cut, Y_cut

def noise_set(Y):    
    """ Generate a new set of images with additive Poisson noise
        
        Args:
            Y (ndarray): set of images where you want to add noise to create an input dataset
            
        Returns:
            X (ndarray): dataset complementary to Y with addictive Poisson noise
            
       Example:
            >>> from dataset_generator import noise_set
            >>> X = noise_set(Y)
    """ 
    
    X=np.empty(Y.shape)
    
    for batch in np.arange(len(Y)):
        for channel in np.arange(len(Y[0])):
            X[batch,channel] = np.random.poisson(Y[batch,channel])    
    return X

def loader_random_generator(X,Y, datasize, batch_size, ROIs = (64,64)):
    """ Generate the dataloader to feed the network with random regions of the image
        
        Args:
            Y (ndarray): target of your dataset --> size: (Batch,Channels,Weight,Height)
            X (ndarray): input of your dataset --> size: (Batch,Channels,Weight,Height)
            datasize = number of random ROIs to generate
            ROIs (tuple): size of random region (ROIs=region of interests)
            batch_size = batch size of the dataloader
            
        Returns:
            Dataloader: format to train the net with a batch size
          
    """ 

    X_tr, Y_tr = one2many_random_ROI(X, Y, datasize, ROIs)

    mu, sigma, X_train, X_train_shape = normalize(X_tr, ROIs=(X_tr.shape[-2],X_tr.shape[-1]))
    Y_train = (split_img(Y_tr, ROIs = (Y_tr.shape[-2],Y_tr.shape[-1])) - mu)/sigma
   
    X_train = np2torch(X_train).reshape(X_train_shape)
    Y_train = np2torch(Y_train).reshape(X_train_shape)

    dataset = TensorDataset(X_train, Y_train)

    return X_train, Y_train, np2torch(mu), np2torch(sigma), DataLoader(dataset, batch_size)

def split_bayer(mosaic):
    """Split a mosaic in its 4 color components
        Args:
            mosaic: 2D array containing a bayer pattern mosaic, e.g. RGBG
        Returns:
            c1: the first color
            c2: the second color
            c3: the third color
            c4: the fourth color
    """
    mosaic = np.tile(mosaic, (4, 1, 1))
    
    mosaic[0, 1::2, 1::2], mosaic[0, 0::2, 1::2], mosaic[0, 1::2, 0::2] = 0,0,0
    mosaic[1, 0::2, 0::2], mosaic[1, 1::2, 1::2], mosaic[1, 1::2, 0::2] = 0,0,0
    mosaic[2, 0::2, 0::2], mosaic[2, 0::2, 1::2], mosaic[2, 1::2, 1::2] = 0,0,0
    mosaic[3, 0::2, 0::2], mosaic[3, 0::2, 1::2], mosaic[3, 1::2, 0::2] = 0,0,0
    
    return mosaic

def rgb2bayer(rgb):
    """Pass from an rgb image to a bayer matrix
        Args:
            rgb: array shape = (weight, height, 3)
        Returns:
            array shape = (weight*2, height*2)
    """
    
    bayer = rgb.repeat(2, axis=0).repeat(2, axis=1)   
    
    mosaic = np.empty((bayer.shape[0],bayer.shape[1]))
    mosaic[0::2, 0::2] = bayer[0::2, 0::2, 0]
    mosaic[1::2, 0::2] = bayer[1::2, 0::2, 1]
    mosaic[0::2, 1::2] = bayer[0::2, 1::2, 1]
    mosaic[1::2, 1::2] = bayer[1::2, 1::2, 2]
    
#     bayer = np.concatenate((bayer, bayer[:,:,1][:,:,np.newaxis]), axis=2)

#     temp = np.empty((4,4, bayer.shape[0],bayer.shape[1]))
#     for i in np.arange(4):    
#         temp[i] = split_bayer(bayer[:,:,i])
#     bayer = temp[0,0] + temp[1,1] + temp[2,2] + temp[3,3]
    
    return mosaic