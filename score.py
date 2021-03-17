import h5py
import netCDF4 as nc
import numpy as np
from sklearn.metrics import confusion_matrix
# from config import config
import os

# fy4afile = os.path.join(config.data_path, 'fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
fy4afile = os.path.join('/data/code-model/cloud/data/','fy4a/20200731/FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20200731033000_20200731033417_4000M_V0001.HDF')
filename = fy4afile.split('/')[-1].split('.')[0]
fy = h5py.File(os.path.join('/tmp', '%s_CLT2.hdf' % filename), 'r')
pred_arr = fy['FY4CLT'][()]
y_pred = pred_arr[80:, 300:-99]
print(y_pred.shape)
# y_pred = pred_arr[99:, 300:-99]
y_pred = y_pred.astype('int16')

h08file = os.path.join('/data/code-model/cloud/data/', 'h08/202007/31/03/NC_H08_20200731_0330_L2CLP010_FLDK.02401_02401.nc')
# h08file = os.path.join(config.data_path, 'h08/202007/31/03/NC_H08_20200731_0330_L2CLP010_FLDK.02401_02401.nc')

ds = nc.Dataset(h08file, 'r')
true_arr = ds.variables['CLTYPE'][:].data
y_true = true_arr[200:1101, 100:1101]
print(y_true.shape)
# y_true[y_true == 2] = 255
# y_true[y_true == 4] = 255
# y_true[y_true == 5] = 255
# y_true[y_true == 8] = 255
y_true[y_true == 0] = 255
y_true[y_true == 10] = 255

cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[1, 3, 6, 7, 9, 255])
# cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 255])
print(cm)
# cm = cm[:5,:5]
# freq = np.sum(cm, axis=1) / np.sum(cm)
freq = np.sum(cm[:-1], axis=1) / np.sum(cm[:-1])
iu = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
FWIoU = (freq[freq > 0] * iu[:-1][freq > 0]).sum()
# FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
print(FWIoU)
