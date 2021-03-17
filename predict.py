import os
import gdal
import numpy as np
import h5py
import torch
import time
import psutil
# psutil是一个跨平台库，能够轻松实现获取系统运行的进程和系统利用率（包括CPU、内存、磁盘、网络等）信息。
# 它主要应用于系统监控，分析和限制系统资源及进程的管理
from datetime import datetime
import netCDF4 as nc
import argparse
from utils.fy4a import AGRI_L1,read_fy4a_arr
from utils.gen_tiles_offs import gen_tiles_offs
from utils.epsg2wkt import epsg2wkt
from utils.preddataset import preddataset,predprefetcher
from modeling.deeplab import *
# from prefetcher import data_prefetcher
# from utils.FWIoU import Frequency_Weighted_Intersection_over_Union

# BLOCK_SIZE = 256
BLOCK_SIZE = 513
OVERLAP_SIZE = 0
geo_range = [5, 54, 70, 139.95, 0.05] 
# geo_range = [5, 54.95, 70, 139.95, 0.05] 
minx = geo_range[2]
maxy = geo_range[1]
res = geo_range[4]
 
fy4a_gt = (minx, res,0.0, maxy, 0.0,(-1)*res)
# 仿射矩阵，左上角像素的大地坐标和像素分辨率（左上角x，x分辨率，仿射变换，左上角y，y分辨率，仿射变换）

# # 各分辨率文件包含的通道号
# CONTENTS = {'0500M': ('Channel02',),
#             '1000M': ('Channel01', 'Channel02', 'Channel03'),
#             '2000M': tuple(['Channel'+"%02d"%(x) for x in range(1, 8)]),
#             '4000M': tuple(['Channel'+"%02d"%(x) for x in range(1, 15)])}
NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
def get_args():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--nproc', type=int, default=psutil.cpu_count(logical=True))    # nproc是操作系统级别对每个用户创建的进程数的限制
    parser.add_argument('--nproc', type=int, default=1)    # nproc是操作系统级别对每个用户创建的进程数的限制
    parser.add_argument('--gpu', type=int, default=torch.cuda.is_available())
    parser.add_argument('--file_list')
    parser.add_argument('--savename', default='res')
    # parser.add_argument('--model', default='/data/code-model/pytorch-deeplab-xception-jpg/run/cloud/deeplab-xception/experiment_3/checkpoint.pth.tar')
    parser.add_argument('--model', default='/home/wzj/pytorch-deeplab-xception-jpg/run/cloud/deeplab-xception/experiment_10/checkpoint.pth.tar')
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--backbone', type=str, default='xception',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')

    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    
    parser.add_argument('--batch-size', type=int, default=50,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='1',    # 0
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
       

    args = parser.parse_args()
    return args

def fy4a_hdf2tif(fy4a_file, fy4a_tif):
    # filename= fy4afile.split('/')[-1].split('.')[0]
    print('the current fy4afile is %s' % fy4a_file)
    data = read_fy4a_arr(fy4a_file, geo_range)
    
    # fydataType = NP2GDAL_CONVERSION[str(data.dtype)]
    # gt = fy4a_gt 
    # dst_nbands = data.shape[0]    # 波段数
    xsize, ysize = data.shape[1:]
    bands_list = []
    for id in range(3):
        array = data[id]
        bands_list.append(array)
    imgarr = np.stack([band for band in bands_list], axis = 2)
    
    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(fy4a_tif, ysize, xsize, 3, 6)
    # dst_ds = driver.Create(dst_file, ysize, xsize, dst_nbands, fydataType)
    # dst_ds.SetGeoTransform(gt)    # 写入仿射变换参数
    # dst_ds.SetProjection(epsg2wkt('EPSG:4326'))    # 写入投影（地图投影信息，字符串表示）
    
    for i in range(3):
        dst_ds.GetRasterBand(i + 1).WriteArray(data[i, :, :])
    del dst_ds

    return xsize, ysize
    
def gen_file_list(geotif):
    file_list = []
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE, OVERLAP_SIZE)
   
    for xoff,yoff in off_list:
        file_list.append((geotif, xoff, yoff))
    return file_list

# %%
# Test the trained model
def predict(args,files_list):
#     with open(args.file_list, 'r') as f:
#         files = f.read().splitlines()
    ds = preddataset(files_list)
    data_loader = torch.utils.data.DataLoader(
        ds, batch_size=args.bs,
        sampler=torch.utils.data.SequentialSampler(ds),
        num_workers=args.nproc, pin_memory=args.gpu, drop_last=False)
    # 初始化参数里有两种sampler：sampler和batch_sampler，都默认为None。
    # 前者的作用是生成一系列的index，而batch_sampler则是将sampler生成的indices打包分组，得到一个又一个batch的index。
    # 例如下面示例中，BatchSampler将SequentialSampler生成的index按照指定的batch size分组。
    
    print('[%s] Start test using: %s.' % (datetime.now(), args.model.split('/')[-1]))
    # datetime.now(tz=None)：返回当前当地时间和日期，如果tz=None
    
    # Define network
    net = DeepLab(num_classes=10,
                    backbone=args.backbone,
                    output_stride=args.out_stride,    # 输入图像的空间分辨率和输出特征图的空间分辨率的比值
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)
    backbone= net.backbone
    # print(backbone.conv1)
    backbone.conv1= nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    # backbone.conv1= nn.Conv2d(14, 64, kernel_size=7, stride=2, padding=3,bias=False)
    # print('change the input channels',backbone.conv1) 
#     checkpoint = torch.load(args.model)
    # m = '/data/code-model/pytorch-deeplab-xception-jpg/run/cloud/deeplab-xception/experiment_3/checkpoint.pth.tar'
    m = '/home/wzj/pytorch-deeplab-xception-jpg/run/cloud/deeplab-xception/experiment_10/checkpoint.pth.tar'
    checkpoint = torch.load(m)
#     args.start_epoch = checkpoint['epoch']
    net = net.cuda()
    if args.gpu:
        checkpoint = torch.load(args.model)
        net.load_state_dict(checkpoint['state_dict'])
        # net.module.load_state_dict(checkpoint['state_dict'])
    else:
        checkpoint = torch.load(args.model,map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'])
    # %%
    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    
#     if args.gpu:
#         net = torch.load(args.model)
#     else:
#         net = torch.load(args.model, map_location=torch.device('cpu'))
#     print('[%s] Model loaded: %s' % (datetime.now(), args.model))

    # start test
    net.eval()
    correct, total = 0, 0
    H=BLOCK_SIZE
    W=BLOCK_SIZE
    pred = torch.empty((len(ds), H, W), dtype=torch.long)    # 创建一个未被初始化数值的tensor, tensor的大小是由size确定
    labels = torch.empty((len(ds), H, W), dtype=torch.long)
    # if args.gpu:
    #     pred = pred.cuda()
    #     labels = labels.cuda()
    prefetcher = predprefetcher(data_loader, args.gpu)
    with torch.no_grad():
#         inputs, targets, index = prefetcher.next()
        inputs, index = prefetcher.next()
        k = 0
#         while (inputs is not None) and (targets is not None):
        while (inputs is not None):
#             if args.bs == 1:.
#                 sampleID = files[index].split('/')[-1].split('.')[0]

            # inputs = torch.tensor(inputs, dtype=torch.float32)
            outputs = net(inputs)  # with shape NCHW
            _, predict = torch.max(outputs.data, 1)  # with shape NHW

#             if args.bs == 1:
#                 print('[%5d/%5d]    %s    test_accu: %.3f' % (total, len(files), sampleID, correct_i/H/W))
#             else:
#                 print('[%5d/%5d] test_accu: %.3f' % (total, len(files), correct_i/H/W/targets.shape[0]))
            
            tile_pred = predict[0]
            tile_pred[tile_pred == 0] = 255
            # tile_pred[tile_pred == 2] = 255
            # tile_pred[tile_pred == 4] = 255
            # tile_pred[tile_pred == 5] = 255
            # tile_pred[tile_pred == 8] = 255
            tile_pred[tile_pred == 10] = 255
            tile_pred = tile_pred.cpu()
            # tile_pred[tile_pred == 0] = 255
            
            pred[index.tolist()] = tile_pred
            # pred[index.tolist()] = torch.tensor(tile_pred, dtype = torch.long)
#             labels[index.tolist()] = targets

            # prefetch train data
#             inputs, targets, index = prefetcher.next()
            inputs, index = prefetcher.next()
            k += 1

    print('[%s] Finished test.' % datetime.now())

    return pred
    
if __name__ == '__main__':
#     predict_cpu()    
    # fy4afile = os.path.join('/data/code-model/cloud/data/','fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
    fy4afile = os.path.join('/home/wzj/cloud_data/predict_result/','FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
    filename = fy4afile.split('/')[-1].split('.')[0]
    # fy4a_tif = '/tmp/test2.tif'
    fy4a_tif = '/home/wzj/cloud_data/predict_result/test.tif'
    # xsize, ysize = fy4a2geotif(fy4afile,fy4a_tif)
    xsize, ysize = fy4a_hdf2tif(fy4afile,fy4a_tif)
    files_list = gen_file_list(fy4a_tif)
    args = get_args()
    pred_list = predict(args,files_list)
    
    pred_arr = np.zeros([xsize,ysize])
    num = len(files_list)
    for i in range(num):
        _,xoff,yoff = files_list[i]
#         tile_pred = predict.cpu().numpy()
        pred_arr[yoff:yoff+BLOCK_SIZE,xoff:xoff+BLOCK_SIZE]=pred_list[i].cpu().numpy()
    
    # dst_hdf = os.path.join('/tmp','%s_CLT2.hdf'%filename)
    dst_hdf = os.path.join('/home/wzj/cloud_data/predict_result/','%s_CLT2.hdf'%filename)
    #HDF5的写入：    
    f = h5py.File(dst_hdf,'w')   #创建一个h5文件，文件指针是f  
    f['FY4CLT'] = pred_arr                 #将数据写入文件的主键'FY4CLT'下面    
    print(pred_arr)
    print(pred_arr.max(),pred_arr.min())
    f.close() 
    
    # dst_file = '/tmp/clt2.tif'
    dst_file = '/home/wzj/cloud_data/predict_result/clt2.tif'
    dataType = NP2GDAL_CONVERSION[str(pred_arr.dtype)]
    gt = fy4a_gt 

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, 1, dataType)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(epsg2wkt('EPSG:4326'))
    dst_ds.GetRasterBand(1).WriteArray(pred_arr)
    
    



    
    
