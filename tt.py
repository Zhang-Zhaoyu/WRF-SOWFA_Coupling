import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt



fileName="/cluster/home/zzhang/sowfa/SOWFA-2.4.0/tools/WRFextraction/wrfout_d01_2015-08-05_05_40_00.nc"



nc_fid = Dataset(fileName,'r')

#print(nc_fid.variables)
#print(nc_fid.variables.keys())


A=np.array([[1, 2, 3], [4, 5, 6]])
B=A*2
#print(B)


time = (nc_fid.variables['XTIME'][:])

time = np.ma.fix_invalid(time)

print('time',time)
print('time.shape',time.shape)


nc_fid = Dataset(fileName,'r')
z = nc_fid.variables['ZNW'][:]
fc = nc_fid.variables['F'][:]
#time = nc_fid.variables['Times'][:]
t0 = time[0]
#print('t0',t0)
print('t0shap',t0.shape)

#t=np.empty(np.size(time))
#for i in np.arange(0,len(time)):
#    t[i]=time[i]-time[0]


#time = [getvar(nc_fid, 'Times')]*24
#t0 = time[0]
# getvar(nc_fid, 'Times')

t = time - t0
#nt = len(t)
#nz = len(z)

#print('time',time)
print('t',t.shape)


print ('ttlala')














