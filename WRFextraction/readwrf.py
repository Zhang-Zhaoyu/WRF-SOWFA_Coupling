from netCDF4 import Dataset
#nc = Dataset(oname, 'w', format='NETCDF4_CLASSIC')
#nc.TITLE = "Extract wind data at 10 m height
#ipath="/sowfa/SOWFA-2.4.0/exampleCases/example.mesoscaleInfluence.SWiFTsiteLubbock.11Nov2013Diurnal"
#fn=ipath+"wrfout_d%02d_"%(domain)+str(year)+"-%02d"%(month)+"-"+"%02d"%(day)+"_%02d"%(hour)+"_%02d_00.nc"%(minute)
#fn=ipath+"wrfout_d%02d_"%(domain)+str(year)+"-%02d"%(month)+"-"+"%02d"%(day)+"_%02d"%(hour)+":%02d:00"%(minute)
fn="wrfout_d01_2015-08-05_05_40_00.nc"
nc_wrf = Dataset(fn, mode="r")
xtimes = nc_wrf.variables['XTIME'] #each time output in wrf nc file
time   = nc_wrf.variables['Times'][:]
cosa    = nc_wrf.variables['COSALPHA'][:,:,:].squeeze()
sina    = nc_wrf.variables['SINALPHA'][:,:,:].squeeze()
lon = nc_wrf.variables['XLONG'][0,:,:]
lat = nc_wrf.variables['XLAT'][0,:,:]
u10r   = nc_wrf.variables['U10'][:,:,:].squeeze()
v10r   = nc_wrf.variables['V10'][:,:,:].squeeze()
u10 = u10r * cosa[0] - v10r * sina[0]
v10 = v10r * cosa[0]  + u10r * sina[0]
