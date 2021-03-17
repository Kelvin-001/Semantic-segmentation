from __future__ import print_function, division
import numpy as np
import pandas as pd
import time, os, sys, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
# from config import config
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
#import torch.nn.functional as functional
import cv2
import tifffile as tiff
import itertools

#import gdal
from osgeo import gdal

from PIL import Image
from mypath import Path
from dataloaders import custom_transforms as tr

random.seed(20201020)

# BLOCK_SIZE = 256
BLOCK_SIZE = 512
OVERLAP_SIZE = 0

# img_path = 'E:/ubuntu_Shared_folder/Fengyun Satellite Competition/cloud_data/img/'
# gt_path = 'E:/ubuntu_Shared_folder/Fengyun Satellite Competition/cloud_data/gt/'
# img_path = '/data/data/cloud_tif/img/'
# gt_path = '/data/data/cloud_tif/gt/'
img_path = '/home/wzj/cloud_data/img_14/'
gt_path = '/home/wzj/cloud_data/gt/'

        
# def gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE):
#     xoff_list = []
#     yoff_list = []
    
#     cnum_tile = int((xsize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
#     rnum_tile = int((ysize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    
#     for j in range(cnum_tile + 1):
#         xoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * j                  
#         if j == cnum_tile:
#             xoff = xsize - BLOCK_SIZE
#         xoff_list.append(xoff)
        
#     for i in range(rnum_tile + 1):
#         yoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * i
#         if i == rnum_tile:
#             yoff = ysize - BLOCK_SIZE
#         yoff_list.append(yoff)
    
#     if xoff_list[-1] == xoff_list[-2]:
#         xoff_list.pop()    # pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值
#     if yoff_list[-1] == yoff_list[-2]:    # the last tile overlap with the last second tile
#         yoff_list.pop()
    
#     return [d for d in itertools.product(xoff_list,yoff_list)]
#     # itertools.product()：用于求多个可迭代对象的笛卡尔积

# def gen_file_list(geotif):
#     file_list = []
#     filename = geotif.split('/')[-1]
#     ds = gdal.Open(geotif)
#     xsize, ysize = ds.RasterXSize, ds.RasterYSize
#     # im = Image.open(geotif).convert('RGB')
#     # xsize, ysize = im.size
#     off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE, OVERLAP_SIZE)
   
#     for xoff, yoff in off_list:    
#         file_list.append((filename, xoff, yoff))     
#     return file_list

# def gen_tile_from_filelist(dir, file_names):
#     files_offs_list=[]
#     for filename in file_names:
#         if filename.endswith(".tif") and filename.split('_')[1].split('.')[0]=='0330' or '0430':
#             file = os.path.join(dir, filename)
#             tif_list = gen_file_list(file)
#             files_offs_list = files_offs_list+tif_list
#     return files_offs_list


class CLOUDdataset(Dataset):

    NUM_CLASSES = 10
    
    def __init__(self, args, files_names, shuffle = False, split = 'train'):
        super().__init__()
        logging.info("ImgloaderPostdam->__init__->begin:")
        random.seed(20201023)

        self.img_size_x = BLOCK_SIZE
        self.img_size_y = BLOCK_SIZE
        self.shuffle = shuffle
        self.file_names = files_names
        self.split = split
        self.args = args
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.sample(range(len(self.file_names)), 1)[0]
        
        filename = self.file_names[idx]
        imgfile = os.path.join(img_path, filename)
        gtfile = os.path.join(gt_path, filename)
        
        imgds = gdal.Open(imgfile,gdal.GA_ReadOnly)
        gtds = gdal.Open(gtfile,gdal.GA_ReadOnly)

        xoff = random.randint(0, 1200 - BLOCK_SIZE)
        yoff = random.randint(0, 981 - BLOCK_SIZE)
        img_tif = imgds.ReadAsArray(xoff, yoff, BLOCK_SIZE, BLOCK_SIZE).astype(np.float32)
        mask = gtds.ReadAsArray(xoff, yoff, BLOCK_SIZE, BLOCK_SIZE)
        
        # tmp = np.zeros(img_tif.shape, dtype=np.float32)
        # img_tif = cv2.normalize(img_tif,tmp,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img_tif[np.isnan(img_tif)] = 0
        imax = img_tif.max()
        imin = img_tif.min()
        img_tif = (img_tif - imin)/(imax - imin)
        img_tif *= 255
        mask[np.isnan(mask)] = 255
        mask[mask == 10] = 0
        mask[mask == 255] = 0
        
        img = torch.from_numpy(img_tif).float()
        mask = torch.from_numpy(mask).float()
         
        if self.split == "train":
            return {'image': img, 'label': mask}
        elif self.split == 'val':
            return {'image': img, 'label': mask}
        #elif self.split == 'test':
        #    return self.transform_ts(sample)
    
    #def transform_tr(self, sample):
    #    composed_transforms = transforms.Compose([
    #        tr.RandomHorizontalFlip(),
    #        tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),    # , fill=255
    #        tr.RandomGaussianBlur(),
    #        #tr.Normalize(mean=(0.001212, 0.001212, 0.001212), std=(0.034795, 0.034795, 0.034795)),
    #        tr.ToTensor()])
    #    return composed_transforms(sample)
    
    #def transform_val(self, sample):
    #    composed_transforms = transforms.Compose([
    #        tr.FixScaleCrop(crop_size=self.args.crop_size),
    #        #tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #        tr.ToTensor()])
    #    return composed_transforms(sample)
    
    #def transform_ts(self, sample):
        #composed_transforms = transforms.Compose([
        #    tr.FixedResize(size=self.args.crop_size),
        #    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #    tr.ToTensor()])
        #return composed_transforms(sample)
    #    return sample
