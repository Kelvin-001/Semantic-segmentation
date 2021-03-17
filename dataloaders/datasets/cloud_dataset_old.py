import numpy as np
import pandas as pd
import time, os, sys, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
# from config import config
import gdal
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

random.seed(20180122)
#np.random.seed(20180122)    
# seed()用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同

BLOCK_SIZE = 256    # 立方体
OVERLAP_SIZE = 0    # 重叠部分

img_path = 'E:/ubuntu_Shared_folder/Fengyun Satellite Competition/fyt_data/img/'
gt_path = 'E:/ubuntu_Shared_folder/Fengyun Satellite Competition/fyt_data/gt/'

num_class = 10

def gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE):
    xoff_list = []
    yoff_list = []
    
    cnum_tile = int((xsize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    rnum_tile = int((ysize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    
    for j in range(cnum_tile + 1):
        xoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * j                  
        if j == cnum_tile:
            xoff = xsize - BLOCK_SIZE
        xoff_list.append(xoff)
        
    for i in range(rnum_tile + 1):
        yoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * i
        if i == rnum_tile:
            yoff = ysize - BLOCK_SIZE
        yoff_list.append(yoff)
    
    if xoff_list[-1] == xoff_list[-2]:
        xoff_list.pop()    # pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值
    if yoff_list[-1] == yoff_list[-2]:    # the last tile overlap with the last second tile
        yoff_list.pop()

    return [d for d in itertools.product(xoff_list,yoff_list)]
    # itertools.product()：用于求多个可迭代对象的笛卡尔积
    
def gen_file_list(geotif):
    file_list = []
    filename = geotif.split('/')[-1]
    ds = gdal.Open(geotif)    # 打开栅格数据集
    # 获取数据集的一些信息
    xsize, ysize = ds.RasterXSize, ds.RasterYSize    # 栅格矩阵的列数，行数
    off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE, OVERLAP_SIZE)
   
    for xoff, yoff in off_list:    
        file_list.append((filename, xoff, yoff))     
    return file_list

def gen_tile_from_filelist(dir, file_names):
    files_offs_list=[]
    for filename in file_names:
        if filename.endswith(".tif") and filename.split('_')[1].split('.')[0]=='0330' or '0430':
            file = os.path.join(dir, filename)
            tif_list = gen_file_list(file)
            files_offs_list = files_offs_list+tif_list
    return files_offs_list

        
class OrigImgdataset(Dataset):
    def __init__(self,  files_names, shuffle=False,normalize=True):
        logging.info("ImgloaderPostdam->__init__->begin:")
        random.seed(20201023)

        self.img_size_x = BLOCK_SIZE
        self.img_size_y =BLOCK_SIZE
        self.shuffle = shuffle
        self.normalize=normalize
#        for i in range(len(files_names)):
#            files_names[i] = list(files_names[i])
#            files_names[i] = tuple(files_names[i])
        self.file_names = files_names
        self.data_set = []
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.sample(range(len(self.file_names)), 1)[0]
        
        filename, xoff, yoff = self.file_names[idx]
        imgfile = os.path.join(img_path, filename)
        gtfile = os.path.join(gt_path, filename)
        imgds = gdal.Open(imgfile,gdal.GA_ReadOnly)
        gtds = gdal.Open(gtfile,gdal.GA_ReadOnly)    # 以只读形式打开
        data_x = imgds.ReadAsArray(xoff, yoff, BLOCK_SIZE, BLOCK_SIZE)    # ReadAsArray支持按块读取影像
        if self.normalize:
            tmp = np.zeros(data_x.shape, dtype=np.float32)
            data_x = cv2.normalize(data_x,tmp,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        data_y = gtds.ReadAsArray(xoff, yoff, BLOCK_SIZE, BLOCK_SIZE)
#        data_x[np.isnan(data_x)] = 0
        data_y[data_y == 10] = 255
        data_y[data_y == 0] = 255
        data_y[np.isnan(data_y)] = 255
        data_x = torch.torch.FloatTensor(data_x)
        data_y = torch.LongTensor(data_y)
        
        return data_x, data_y