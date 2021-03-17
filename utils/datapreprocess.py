import numpy as np
# from fy4a import FY4A_AGRI_L1
# from config import config
# import netCDF4 as nc
import xarray as xr
from h8_find_fy4a import h8_find_fy4a
import os, sys
import gdal
import osr
from epsg2wkt import epsg2wkt
from fy4a import AGRI_L1,read_fy4a_arr
from gen_tiles_offs import gen_tiles_offs
import time
import cv2
from PIL import Image
import tifffile

out_georange = [5, 54, 80, 139.95, 0.05]
out_gt = (out_georange[2], out_georange[4], 0.0, out_georange[1], 0.0, (-1) * out_georange[4])

out_crs = 'EPSG:4326'
out_proj = epsg2wkt('EPSG:4326')

# 数据根目录
fy4a_root = os.path.join('/data/data/meteo','fy4a')
h08_root = os.path.join('/data/data/meteo','h08')
fy4a_tif = '/data/data/cloud_tif/img'
h08_tif = '/data/data/cloud_tif/gt'

# fy4a_tif = '/tmp/'
# h08_tif = '/tmp/'

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

def preprocessfy4a(fy4a_file, fy4a_dst):
    print('the current fy4afile is %s' % fy4a_file)
    data = read_fy4a_arr(fy4a_file, out_georange)
    
    xsize = data.shape[1]
    ysize = data.shape[2]
    
    fydataType = NP2GDAL_CONVERSION[str(data.dtype)]
    dst_nbands = data.shape[0]

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(fy4a_dst, ysize, xsize, dst_nbands, fydataType)
    dst_ds.SetGeoTransform(out_gt)
    dst_ds.SetProjection(out_proj)
    
    if dst_nbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(data)
    else:
        for i in range(dst_nbands):
            dst_ds.GetRasterBand(i + 1).WriteArray(data[i, :, :])
    del dst_ds

def preprocessh08(h08file, h08_dst):
    try:
        ds = xr.open_dataset(h08file)
    except:
        print('error')
        sys.exit()
    cltype = ds.CLTYPE.data
    Lon = ds.longitude.data
    Lat = ds.latitude.data
    LonMin, LatMax, LonMax, LatMin = [Lon.min(), Lat.max(), Lon.max(), Lat.min()] 
    
    # 分辨率计算
    N_Lat = len(Lat) 
    N_Lon = len(Lon)
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)
    
    sxoff = int(round((out_georange[2]-LonMin)/Lon_Res))
    syoff = int(round((LatMax-out_georange[1])/Lat_Res))
    
    exoff = int(round((out_georange[3]-LonMin)/Lon_Res))+1
    eyoff = int(round((LatMax-out_georange[0])/Lat_Res))+1
    
    if Lon_Res==out_georange[4] and Lat_Res==out_georange[4]:
        data = cltype[syoff:eyoff,sxoff:exoff]
    else:
        # resolution is not the same as the output requirement, need to be tested
        lats = np.arange(out_georange[0], out_georange[1]+out_georange[4], out_georange[4])
        lons = np.arange(out_georange[2], out_georange[3]+out_georange[4], out_georange[4])
        temp = cltype[syoff:eyoff,sxoff:exoff]
        data = temp.interp(latitude=lats, longitude=lons, method='nearest')
    
    ysize = data.shape[0]
    xsize = data.shape[1]
    
    dataType = NP2GDAL_CONVERSION[str(data.dtype)]

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(h08_dst, xsize, ysize, 1, dataType)
    dst_ds.SetGeoTransform(out_gt)
    dst_ds.SetProjection(out_proj)
    
    dst_ds.GetRasterBand(1).WriteArray(data)
    del dst_ds
    print('the current h08file is %s' % h08file)

def gen_file_list(geotif):
    file_list = []
    filename = geotif.split('/')[-1]

    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE)
   
    for xoff,yoff in off_list:    
        file_list.append((filename, xoff, yoff))     
    return file_list
def fy4a_hdf2tif(fy4a_file, fy4a_tif):
    print('the current fy4afile is %s' % fy4a_file)
    data = read_fy4a_arr(fy4a_file, out_georange)
    xsize,ysize = data.shape[1:]
    bands_list = []
    for id in range(3):
        array = data[id]

        bands_list.append(array)
    imgarr = np.stack([band for band in bands_list],axis=2)
    
    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(fy4a_tif, ysize, xsize, 3, 6)
    
    for i in range(3):
        dst_ds.GetRasterBand(i + 1).WriteArray(data[i, :, :])
    del dst_ds
    
    # tmp = tifffile.imread(fy4a_tif)
    # assert tmp.all()==imgarr.all()
    
def h08_nc2tif(h08file, h08jpg):
    try:
        ds = xr.open_dataset(h08file)
    except:
        print('error')
        sys.exit()
    cltype = ds.CLTYPE.data
    Lon = ds.longitude.data
    Lat = ds.latitude.data
    LonMin, LatMax, LonMax, LatMin = [Lon.min(), Lat.max(), Lon.max(), Lat.min()] 
    
    # 分辨率计算
    N_Lat = len(Lat) 
    N_Lon = len(Lon)
    Lon_Res = (LonMax - LonMin) / (float(N_Lon) - 1)
    Lat_Res = (LatMax - LatMin) / (float(N_Lat) - 1)
    
    sxoff = int(round((out_georange[2]-LonMin)/Lon_Res))
    syoff = int(round((LatMax-out_georange[1])/Lat_Res))
    
    exoff = int(round((out_georange[3]-LonMin)/Lon_Res))+1
    eyoff = int(round((LatMax-out_georange[0])/Lat_Res))+1
    
    if Lon_Res==out_georange[4] and Lat_Res==out_georange[4]:
        data = cltype[syoff:eyoff,sxoff:exoff]
        data = data.astype('int8')
            
        im = Image.fromarray(data)
        im =im.convert("L")
        im.save(h08jpg)
        # tmp = cv2.imread(h08jpg,-1)
        # assert tmp.all()==data.all()

def main():   
    tiles_list = []
    start = time.time()
    for root, dirs, files in os.walk(h08_root):
        files = list(filter(lambda x: x.endswith(".nc"), files))
        for file in files:
            h08date = file.split('_')[2]
            h08time = file.split('_')[3]
            hour = h08time[:2]
            minu = int(h08time[2:])
            if int(hour) >= 3 and int(hour) <= 4:
#             if int(hour) >= 1 and int(hour) <= 8:
        #                 daytime in China
                if minu == 30: 
#                 if minu == 20 or minu == 30 or minu == 40 or minu == 50:  
        #                     have corresponding fy4a data
                    h08_file = os.path.join(root, file)
                    fy4a_file = h8_find_fy4a(h08_file, fy4a_root) 
                    if os.path.exists(fy4a_file): 
#                         h08_dst = os.path.join(h08_tif, h08date + '_' + h08time + '.tif')
#                         fy4a_dst = os.path.join(fy4a_tif, h08date + '_' + h08time + '.tif')
#                         preprocessh08(h08_file, h08_dst)
#                         preprocessfy4a(fy4a_file, fy4a_dst)
                        fy4a_dst = os.path.join(fy4a_tif, h08date + '_' + h08time + '.tif')
                        h08_dst = os.path.join(h08_tif, h08date + '_' + h08time + '.tif')
                        h08_nc2tif(h08_file, h08_dst)
                        fy4a_hdf2tif(fy4a_file,fy4a_dst)
#                         tif_list = gen_file_list(h08_dst)
#                         tiles_list.append(tif_list)
#                         try:
#                             preprocessh08(h08_file, h08_dst)
#                             preprocessfy4a(fy4a_file, fy4a_dst)
#                         except:
#                             continue
#                         else:
#                             tif_list = gen_file_list(h08_dst)
#                             tiles_list.append(tif_list)
    
    file = open('/tmp/files_list.txt','w');
    file.write(str(tiles_list));
    file.close();
    
    end = time.time()
    period = end - start
    print('the total time is %s seconds' % period)

if __name__ == '__main__':
    main()
#     fy4afile = os.path.join(config.data_path,'fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
#     filename = fy4afile.split('/')[-1].split('.')[0]
#     
#     fy4a_dst = '/tmp/2020fy.tif'
#     preprocessfy4a(fy4afile, fy4a_dst)
    
#     h08file = '/tmp/NC_H08_20200731_0330_L2CLP010_FLDK.02401_02401.nc'
#     h08_dst = '/tmp/2020h08.tif'
#     preprocessh08(h08file, h08_dst)