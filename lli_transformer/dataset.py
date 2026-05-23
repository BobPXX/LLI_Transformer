import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2
import torchvision
import random

#-------------------------------------------------------------
#load psf
def load_psf(psf_dir,mask_size=190):
    mask = np.load(psf_dir)
    pad=int(mask_size/2)
    mask = cv2.resize(mask, (mask_size, mask_size))
    psf0 = np.pad(mask[:,:,0], ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    psf1 = np.pad(mask[:,:,1], ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    psf2 = np.pad(mask[:,:,2], ((pad, pad), (pad, pad)), 'constant', constant_values=(0, 0))
    psf0_F = np.fft.fft2(psf0)
    psf1_F = np.fft.fft2(psf1)
    psf2_F = np.fft.fft2(psf2)
    return psf0_F, psf1_F, psf2_F
#----------------------------------------------------------------
def create_pattern(mask_size, image_dir, psf0_F, psf1_F, psf2_F,train):
    image=cv2.imread(image_dir)
    if train:
        if random.randint(0, 1) > 0.5:
            image = cv2.flip(image, 1)

    image = cv2.resize(image, (mask_size, mask_size))

    pad = int(mask_size / 2)
    channel0 = image[:, :, 0]
    channel0 = np.pad(channel0, ((pad,pad), (pad,pad)), 'constant', constant_values=(0, 0))
    pattern0 = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(channel0)) * np.fft.fftshift(psf0_F))))

    channel1 = image[:, :, 1]
    channel1 = np.pad(channel1, ((pad,pad), (pad,pad)), 'constant', constant_values=(0, 0))
    pattern1 = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(channel1)) * np.fft.fftshift(psf1_F))))

    channel2 = image[:, :, 2]
    channel2 = np.pad(channel2, ((pad,pad), (pad,pad)), 'constant', constant_values=(0, 0))
    pattern2 = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(np.fft.fft2(channel2)) * np.fft.fftshift(psf2_F))))

    pattern=np.dstack((pattern0, pattern1, pattern2))

    return pattern


def image_process(image_dir,psf0_F, psf1_F, psf2_F,train, mask_size=190, output_size=224, num_channels=3):
    pattern=create_pattern(mask_size,image_dir,psf0_F, psf1_F, psf2_F,train)
    '''
    if train==True:
        #ori image size is 380x380
        l =random.randint(145,155)  # crop size is (290,310), crop_size/2=(145,155)
        yc=random.randint(185,195) #center is 190, (185,195)
        xc=random.randint(185,195)
        pattern=pattern[yc-l:yc+l,xc-l:xc+l]
    else:
        pattern= pattern[40:340,40:340] #(300,300)
    '''
    pattern = pattern[40:340, 40:340]

    pattern = cv2.resize(pattern, (output_size, output_size), interpolation=cv2.INTER_AREA)
    pattern = (pattern - pattern.mean()) / pattern.std()
    #image = feature.local_binary_pattern(image, 8, 1, method="uniform")
    #image=cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX)
    return np.reshape(pattern, (num_channels,output_size, output_size))

class get_data(Dataset):
    def __init__(self, train, psf_dir, filename_list_dir, label_list_dir):
        self.train=train
        self.psf_dir=psf_dir
        self.filename_list = np.load(filename_list_dir)
        #self.filename_list=self.filename_list[0:50]
        self.label_list = np.load(label_list_dir)
        #self.label_list = self.label_list[0:50]
        self.psf0_F, self.psf1_F, self.psf2_F=load_psf(psf_dir=self.psf_dir)

    def __len__(self):
        return len(self.filename_list)

    def __getitem__(self, idx):
        pattern = image_process(train=self.train,image_dir=self.filename_list[idx],psf0_F=self.psf0_F, psf1_F=self.psf1_F, psf2_F=self.psf2_F)
        label=self.label_list[idx]
        return pattern, label
