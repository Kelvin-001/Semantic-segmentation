# -*- coding: utf-8 -*-
"""
从FY-4A标称数据提取指定范围指定通道

@Time    : 2018/11/14 12:46:47
@Author  : modabao
"""

import xarray as xr
import numpy as np
import sys
from utils.projection import latlon2linecolumn


# 各分辨率文件包含的通道号
CONTENTS = {'0500M': ('Channel02',),
            '1000M': ('Channel01', 'Channel02', 'Channel03'),
            '2000M': tuple(['Channel'+"%02d"%(x) for x in range(1, 8)]),
            '4000M': tuple(['Channel'+"%02d"%(x) for x in range(1, 15)])}
# 各分辨率全圆盘数据的行列数
SIZES = {'0500M': 21984,
         '1000M': 10992,
         '2000M': 5496,
         '4000M': 2748}


class AGRI_L1(object):
    """
    FY4A AGRI LEVEL1数据
    """
    def __init__(self, file_path, geo_desc=None):
        """
        获得L1数据hdf5文件对象
        """
#         self.file_path = file_path
#         self.geo_desc = geo_desc
        try:
            self.dataset = xr.open_dataset(file_path)
        except OSError: 
            print('No such file or directory')
            self.dataset =None
            
        else:
            self.resolution = file_path[-15:-10]
            self.line_begin = self.dataset.attrs['Begin Line Number']
            self.line_end = self.dataset.attrs['End Line Number']
            self.column_begin = self.dataset.attrs['Begin Pixel Number']
            self.column_end = self.dataset.attrs['End Pixel Number']
            self.set_geo_desc(geo_desc)
    
#     def readfile(self):
#         self.dataset = xr.open_dataset(file_path)
        

    def __del__(self):
        """
        确保关闭L1数据hdf5文件
        """
#         self.dataset.close()
        if self.dataset is None:
            sys.exit(1)
        else:
            self.dataset.close()
        
    def set_geo_desc(self, geo_desc):
        if geo_desc is None:
            self.line = self.column = self.geo_desc = None
            return
        # 先乘1000取整是为了减少浮点数的精度误差累积问题
        lat_S, lat_N, lon_W, lon_E, step = [1000 * x for x in geo_desc]    # np.arange():函数返回一个有终点和起点的固定步长的排列
        lat = np.arange(lat_N, lat_S-1, -step) / 1000
        lon = np.arange(lon_W, lon_E+1, step) / 1000
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)    # 生成网格点坐标矩阵
        # 求geo_desc对应的标称全圆盘行列号
        line, column = latlon2linecolumn(lat_mesh, lon_mesh, self.resolution)
        self.line = xr.DataArray(line, coords=(('lat', lat), ('lon', lon)), name='line')    # xarray.DataArray 是一个使用标签的多维数组
        self.column = xr.DataArray(column, coords=(('lat', lat), ('lon', lon)), name='column')
        self.geo_desc = geo_desc

    def extract(self, channel_name, calibration='reflectance',
                geo_desc=None, interp_method='nearest'):
        """
        按通道名和定标方式提取geo_desc对应的数据
        channel_name：要提取的通道名（如'Channel01'）
        
        calibration: {'dn', 'reflectance', 'radiance', 'brightness_temperature'}
        """
        # calibration:校准  dn:数字量化值（像素值的通用术语是数字量化值或DN值，它通常被用来描述还没有校准到具体意义单位的像素值）
        # reflectance：反射率   radiance：辐射亮度   brightness_temperature：亮温
        if geo_desc and geo_desc != self.geo_desc:
            self.set_geo_desc(geo_desc)
#         dn_values = self.dataset[f'NOM{channel_name}']
        dn_values = self.dataset['NOM%s'%channel_name]
        dtype = dn_values.dtype
        dn_values = dn_values.rename({dn_values.dims[0]: 'line', dn_values.dims[1]: 'column'})
        dn_values = dn_values.assign_coords(line=range(self.line_begin, self.line_end+1),
                                            column=range(self.column_begin, self.column_end+1))
        if self.geo_desc:
            # 若geo_desc已指定，则插值到对应网格
            dn_values = dn_values.interp(line=self.line, column=self.column, method=interp_method)
            del dn_values.coords['line'], dn_values.coords['column']
        else:
            # 若geo_desc为None，则保留原始NOM网格
            pass
        return self.calibrate(channel_name, calibration, dn_values)
#         data = dn_values.data
# #         data.dtype=dtype
#         return data

    def calibrate(self, channel_name, calibration, dn_values):
        """
        前面6个通道，用查找表和系数算出来都是反射率，后面用查找表是亮温，用系数是辐射度。
        """
        if calibration == 'dn':
            dn_values.attrs = {'units': 'DN'}
            return dn_values
        channel_num = int(channel_name[-2:])
        dn_values = dn_values.fillna(dn_values.FillValue)  # 保留缺省值
        if ((calibration == 'reflectance' and channel_num <= 6) or
            (calibration == 'radiance' and channel_num > 6)):
            k, b = self.dataset['CALIBRATION_COEF(SCALE+OFFSET)'].values[channel_num-1]
            data = k * dn_values.where(dn_values != dn_values.FillValue) + b
            data.attrs['units'] = '100%' if calibration == 'reflectance' else 'mW/ (m2 cm-1 sr)'
        elif calibration == 'brightness_temperature' and channel_num > 6:
#             cal_table = self.dataset[f'CAL{channel_name}']
            cal_table = self.dataset['CAL%s'%channel_name]
            cal_table = cal_table.swap_dims({cal_table.dims[0]: 'dn'})
            data = cal_table.interp(dn=dn_values)
            del data.coords['dn']
            data.attrs = {'units': 'K'}
        else:
            raise ValueError('%s没有%s的定标方式'%(channel_name,calibration))
        data.name = '%s_%s'%(channel_name,calibration)
        return data
    
def read_fy4a_arr(h5name,geo_range):
# geo_desc = [5, 54.95, 70, 139.95, 0.05]  # 顺序为南、北、西、东、分辨率，即[lat_s, lat_n, lon_w, lon_e, resolution]

    file = AGRI_L1(h5name, geo_range)
    if file.dataset is None:
        print('error opening the file %s'%h5name)
        sys.exit(1)
    channels = CONTENTS['4000M']
    bands_list=[]
    for channel in channels:
        channel_num = int(channel[-2:])
        if channel_num<=6:
            bands_list.append(file.extract(channel))
        if channel_num>6:
            bands_list.append(file.extract(channel, calibration='brightness_temperature'))
        
    imgarr = np.stack(band for band in bands_list)
#     print(imgarr.shape)    
       
    return imgarr
    
if __name__ == '__main__':
    fy4afile = '/mnt/win/code/dataservice/cloud/data/fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF'
    filename= fy4afile.split('/')[-1].split('.')[0]
#     print('the current fy4afile is %s'%fy4afile)
    geo_range = [5, 54.95, 70, 139.95, 0.05] 
    
    file = AGRI_L1(fy4afile, geo_range)
    channels = CONTENTS['4000M']
    bands_list=[]
    for channel in channels:
        channel_num = int(channel[-2:])
        if channel_num<=6:
            bands_list.append(file.extract(channel))
        if channel_num>6:
            bands_list.append(file.extract(channel, calibration='brightness_temperature'))
        
    imgarr = np.stack(band for band in bands_list)    # np.stack(arrays, axis = 0):stack函数用于堆叠数组，其中arrays是需要进行堆叠的数组，axis是堆叠时使用的轴
    print(imgarr.shape)