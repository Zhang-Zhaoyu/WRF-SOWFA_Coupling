#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Matt Churchfield
# National Rewable Energy Laboratory
# 8 October, 2016
#
# Cp                            Constant pressure specific heat of air.
# R                             Specific gas constant of air.
# nHrs                          Number of hours of WRF data to read in.
# nPerHr                        Number of outputs per hour of WRF data.
# nLevels                       Number of contour levels on contour plots.
# fileName                      Root file name for .nc input files.
# fieldVarName                  List of field variable names to read.
# multiplyByfc                  Flag to multiply variable by f_Coriolis or not.
# timeAvgWindowSize             Size of time sample window for time averaging.
# surfaceVarName                List of surface variables to read.




# Load important packages
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from datetime import timedelta
import datetime
import pandas as pd


# User input
Cp = 1005.0
R = 286.9
p0 = 100000.0
kappa = 2.0/7.0
nHrs = 0.5 
nPerHr = 6
nLevels = 41

fileName="/cluster/home/zzhang/sowfa/SOWFA-2.4.0/tools/WRFextraction/wrfout_d01_2015-08-05_05_40_00.nc"
#fileName = '../tendency_ncfiles/SWIFT_all_w0_L19000.nc'
#fileName = '../tendency_ncfiles/SWIFT_all_w60_L0.nc'
#fileName = '../tendency_ncfiles/SWIFT_all_w60_L19000.nc'
#fileName = '../tendency_ncfiles/SWIFT_all_neutral_FNL_notdivbyfc_tend_L4000'
#fileName = '.\TendencyForcing/tendency_ncfiles\SWIFT_all_neutral_FNL_notdivbyfc_1gp_tend_L0'

fieldVarName = ['U','V','T','Uadv','Vadv','Thadv','Ug','Vg','Utend','Vtend','Ucor','Vcor','Uphys','Vphys']
# Th to T /

multiplyByfc = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
timeAvgWindowSize = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];

surfaceVarName = ['HFX','LH','TSK','T2','PSFC']

writeFields = True
writeForcings = True
writeSurfaceValues = True
writeProbes = True
probeX = [2500.0, 2500.0, 2500.0, 2500.0, 2500.0]
probeY = [ 500.0, 1500.0, 2500.0, 3500.0, 4500.0]
probeMinZ = 5.0
probeMaxZ = 1995.0
probedZ = 10.0
writeInitialValues = True
initialTime = 12.0

plotFields = False
plotSurface = True



# Declare some variables
U = []
V = []
T = []
Uadv = []
Vadv = []
Thadv = []
Ug = []
Vg = []
Utend = []
Vtend = []
Ucor = []
Vcor = []
Uphys = []
Vphys = []
Qs = []
Ts = []
T2m = []


# zzy changed variables z-ZNW fc-F  time-XTIME Psfc-PSFC 
nc_fid = Dataset(fileName,'r')

z = nc_fid.variables['ZNW'][:]
print('z',z)

fc = nc_fid.variables['F'][:]
#print('fc',fc)

# zzy np.array: remove mask
time1 =  np.array(nc_fid.variables['Times'][:]) # zzy *24:day to hour

# zzy from bytes to string
time2 = time1.astype('U1') #.reshape(time.size,-1)
time2 = np.array(time2)

# zzy concatenate strings in every rows
time3=["" for x in range(len(time2))]
for i in range(len(time2)):
   time3[i]="".join(time2[i,:])
time=np.reshape(time3,(len(time2),1))

#print('time',time)
#print('time.shape',time.shape)

# zzy np.array string to datetime type
timestamp=["" for x in range(len(time))]

for i in range(len(time)):
   timestamp[i]=[pd.to_datetime(time[i],errors='coerce',format='%Y-%m-%d_%H:%M:%S')]

# zzy datetime force to array
time4=np.asarray(timestamp)

#print('time4',time4)
#print('time4[1]',time4[1])

# zzy array 3D to 2D / 2D to 1D
time_2d=np.reshape(np.transpose(time4,(2,0,1)),(-1,1))
time_1d=np.ravel(time_2d)

#print('time_1d',time_1d)
#print('time_1d[1]',time_1d[0])

# zzy datetime string to float
# zzy define conversion function
def datetime2matlabdn(dt):
    ord = datetime.date.toordinal(dt)
    mdn = dt + timedelta(days = 366)
    frac = (dt-datetime.datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    return datetime.date.toordinal(mdn) + frac

# zzy test defined function
#print('1')

# zzy apply the definde fuction
time=[]
for i in range(len(timestamp)):
   time.append(datetime2matlabdn(pd.to_datetime(time_1d[i])))
time=np.asarray(time)*24	# zzy *24:day to hour      

#print('time',time)
#print('timediff',time-time[0])

############

t0 = time[0]
#print('t0',t0)

t = time - t0
#print('t.shape',t.shape)

nt = len(t)
nz = 87 # zzy len(z)


# Loop through the field variables.
for m in range(len(fieldVarName)):

   print('m_count',m)
    
   var = np.zeros([nz,nt])
   ncVar = nc_fid.variables[fieldVarName[m]][:]
   # zzy ncvar.shape (354, 87, 108, 100)
   # zzy ( Time, bottom_top, south_north, west_east_stagger )
   
   # zzy transpose!!!!
   var =np.transpose(ncVar)


   # zzy var.shape (100, 108, 87, 354)
   # zzy ( west_east_stagger, south_north, bottom_top, Time )
 
   # If the variable needs to be multiplied by f_Coriolis, do that now.
   if (multiplyByfc[m] == 1.0):
      var = var * fc

   var =np.transpose(var)

   # Apply time averaging to the variables.   
   
   varAvg = np.zeros([nt,nz])	#zzy size not match     
   #varAvg = np.zeros(100,87,354)
   
   print('nz',nz)
   print('nt',nt)
   print('var',var)
   print('var.shape',var.shape)
   print('testlala',var[:,:,1,1])
   #print('fc',fc)   
   #print('fc.shape',fc.shape)
   #print('m=',m)
   #test2=np.mean( var[:,0:0] , axis=1 )
   #print('test2',test2)
   #print('test2.shape',test2.shape)

   #test3=var[:,0:0]   
   #print('test3',test3)
   #print('test3.shape'.test3.shape)


   for i in range(nt):
      #print('i_count',i)
      for n in range(nz):
         iMin = max(0, i - (timeAvgWindowSize[m]-1)/2)
         iMax = min(nt,i + (timeAvgWindowSize[m]-1)/2)
         #varAvg[:,i] =np.mean( var[:,int(iMin):int(iMax+1)] , axis=1 )  
         varAvg[i,n] = var[i,n].mean(axis=(0,1))
         #print('n_count',n)
   arAvg=np.transpose(varAvg)


 # zzy changed from: var[:,iMin):iMax+1].mean(axis=1)     # zzy float to int
      
   if (fieldVarName[m] == 'U'):
       U = varAvg;	#zzy x-velocity
   elif (fieldVarName[m] == 'V'):
       V = varAvg;	#zzy y-velocity
   elif (fieldVarName[m] == 'T'):	#zzy Th to T
       Th = varAvg;	#zzy potential temperature
   elif (fieldVarName[m] == 'Uadv'):
       Uadv = varAvg;	#zzy x advective wind
   elif (fieldVarName[m] == 'Vadv'):
       Vadv = varAvg;	#zzy y advective wind
   elif (fieldVarName[m] == 'Thadv'):	
       Thadv = varAvg;	#zzy potential temperature advective wind
   elif (fieldVarName[m] == 'Ug'):
       Ug = varAvg;	#zzy x pressure gradient wind components
   elif (fieldVarName[m] == 'Vg'):
       Vg = varAvg;	#zzy y pressure gradient wind components
   elif (fieldVarName[m] == 'Utend'):
       Utend = varAvg;  #zzy x wind-speed tendency
   elif (fieldVarName[m] == 'Vtend'):
       Vtend = varAvg;	#zzy y wind-speed tendency
   elif (fieldVarName[m] == 'Ucor'):
       Ucor = varAvg;	#zzy x Coriolis force
   elif (fieldVarName[m] == 'Vcor'):
       Vcor = varAvg;	#zzy y Coriolis force
   elif (fieldVarName[m] == 'Uphys'):
       Uphys = varAvg;	#zzy x turbulent diffusion wind components
   elif (fieldVarName[m] == 'Vphys'):
       Vphys = varAvg;	#zzy y turbulent diffusion wind components


   # Plot the field variable time-height history.
   if (plotFields):
      fig = plt.figure(m+1)
      cs = plt.contourf(t,z,varAvg,nLevels,cmap=plt.cm.Spectral_r)
      cxl = plt.xlabel('t (hr)')
      cyl = plt.ylabel('z (m)')
      ax = plt.gca()
      ax.set_ylim([0,2000])
      cbar = plt.colorbar(cs, orientation='vertical')
      cbar.set_label("%s (%s)" % (fieldVarName[m],nc_fid.variables[fieldVarName[m]].units))
      plt.show()
      


hInd = 100
U_h = U[hInd,:]
V_h = V[hInd,:]
U_derived = np.zeros(U_h.shape)
V_derived = np.zeros(V_h.shape) 
U_derived_tend = np.zeros(U_h.shape)
V_derived_tend = np.zeros(V_h.shape)
for i in range(nt):
    if (i == 0):
        U_derived[i] = U_h[i]
        V_derived[i] = V_h[i]
        U_derived_tend[i] = U_h[i]
        V_derived_tend[i] = V_h[i]
    else:
        dt = (t[i] - t[i-1])*3600.0
        U_derived[i] = U_derived[i-1] + dt*(Uadv[hInd,i]-Vg[hInd,i]+Ucor[hInd,i]+Uphys[hInd,i])
        V_derived[i] = V_derived[i-1] + dt*(Vadv[hInd,i]+Ug[hInd,i]+Vcor[hInd,i]+Vphys[hInd,i])
        U_derived_tend[i] = U_derived_tend[i-1] + dt*(Utend[hInd,i])
        V_derived_tend[i] = V_derived_tend[i-1] + dt*(Vtend[hInd,i])
        

fig = plt.figure(98)
plt.plot(t,U_h,'b-')
plt.plot(t,U_derived,'b--')
plt.plot(t,U_derived_tend,'b:')

fig = plt.figure(99)
plt.plot(t,V_h,'r-')
plt.plot(t,V_derived,'r--')
plt.plot(t,V_derived_tend,'r:')


     

# Write out the field files
if (writeFields):
   fid = open('fieldTable','w')
     
   # Write the height list for the momentum fields
   fid.write('sourceHeightsMomentum\n')    
   fid.write('(\n')
   for j in range(nz):
      fid.write('    ' + str(z[j]) + '\n')
   fid.write(');\n\n')
         
   # Write the x-velocity
   fid.write('sourceTableMomentumX\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         fid.write(str(U[j,n]) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
         
   # Write the y-velocity
   fid.write('sourceTableMomentumY\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         fid.write(str(V[j,n]) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
         
   # Write the z-velocity (hard coded to 0.0 zero here)
   fid.write('sourceTableMomentumZ\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         fid.write(str(0.0) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
     
   # Write the height list for the temperature fields
   fid.write('sourceHeightsTemperature\n')    
   fid.write('(\n')
   for j in range(nz):
      fid.write('    ' + str(z[j]) + '\n')
   fid.write(');\n\n')
         
   # Write the temperature
   fid.write('sourceTableTemperature\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         fid.write(str(Th[j,n]) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
    
   fid.close()
   
   
   
if (writeForcings):
   fid = open('forcingTable','w')
     
   # Write the height list for the momentum forcings
   fid.write('sourceHeightsMomentum\n')    
   fid.write('(\n')
   for j in range(nz):
      fid.write('    ' + str(z[j]) + '\n')
   fid.write(');\n\n')
         
   # Write the x-momentum forcing
   fid.write('sourceTableMomentumX\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         Uforcing = Uadv[j,n] - Vg[j,n] 
         fid.write(str(Uforcing) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
         
   # Write the y-momentum forcing
   fid.write('sourceTableMomentumY\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         Vforcing = Vadv[j,n] + Ug[j,n]
         fid.write(str(Vforcing) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
         
   # Write the z-momentum forcing (hard coded to 0.0 zero here)
   fid.write('sourceTableMomentumZ\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         fid.write(str(0.0) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
     
   # Write the height list for the temperature forcing
   fid.write('sourceHeightsTemperature\n')    
   fid.write('(\n')
   for j in range(nz):
      fid.write('    ' + str(z[j]) + '\n')
   fid.write(');\n\n')
         
   # Write the temperature forcing
   fid.write('sourceTableTemperature\n')    
   fid.write('(\n')
   for n in range(nt):
      fid.write('    (' + str(t[n]*3600) + ' ')     
      for j in range(nz):
         fid.write(str(Thadv[j,n]) + ' ')
      fid.write(')\n')
   fid.write(');\n\n')
    
   fid.close()
   
   

if (writeInitialValues):
   ind = (np.abs(t-initialTime)).argmin()
   fid = open('initialValues','w')   
   for j in range(nz):
      fid.write('    (' + str(z[j]) + ' ' + str(U[j,ind]) + ' ' + str(V[j,ind]) + ' ' + str(Th[j,ind]) + ')\n')
   fid.close()
   
   
   
if (writeProbes):
   tol = 0.001
   probeZ = np.linspace(probeMinZ, probeMaxZ, (probeMaxZ-probeMinZ)/probedZ + 1)
   for i in range(len(probeX)):
      probeFileName = 'probe' + str(i+1)
      fid = open(probeFileName,'w')
      for j in range(len(probeZ)):
         fid.write('                   (' + str(probeX[i]+tol) + ' ' + str(probeY[i]+tol) + ' ' + str(probeZ[j]+tol) +')\n')
      fid.close()
   
   
   
# Now loop through the surface variables.
pS = np.zeros([nt])
TS = np.zeros([nt])
T2m = np.zeros([nt])
qS = np.zeros([nt])

for m in range(len(surfaceVarName)):
    
   # Read in the data file for each hour. 
   var = np.zeros([nz,nt])
   nc_fid = Dataset(fileName, 'r') 
   ncVar = nc_fid.variables[surfaceVarName[m]][:]
   var = np.transpose(ncVar)
      
   if (m == 0):
      pI = nc_fid.variables['PSFC'][:]
      TI = nc_fid.variables['TSK'][:]
      T2I = nc_fid.variables['T2'][:]
      pS = np.transpose(pI)
      TS = np.transpose(TI)
      T2m = np.transpose(T2I)

     
      
   # Plot the variable's time history.
   if (plotSurface):
      fig = plt.figure()
      cs = plt.plot(t,var,'r-')
      cxl = plt.xlabel('t (hr)')
      cyl = plt.ylabel("%s (%s)" % (surfaceVarName[m],nc_fid.variables[surfaceVarName[m]].units))
   
   # Treat surface heat flux specially--compute surface temperature flux.  Here
   # Qs = hfx/(Cp*rho_s), where rho_s = p_s/(R*T_2m).  I used the 2m temperature
   # instead of the skin temperature.
   if (surfaceVarName[m] == 'HFX'):
       qS = np.zeros([nt])
       rhoS = np.zeros([nt])
       for i in range(nt):
           rhoS[i] = pS[i]/(R * T2m[i])
           qS[i] = var[i]/(Cp * rhoS[i])
    
       if (plotSurface):
          fig = plt.figure()
          cs = plt.plot(t,qS,'r-')
          cxl = plt.xlabel('t (hr)')
          cyl = plt.ylabel('Qs (K m/s)')
          



# T_S and T_2m are real temperature, but we also want these in potential temperature.
theta2m = np.zeros([nt])
thetaS = np.zeros([nt])
for i in range(nt):
   thetaS[i] = TS[i]*((p0/pS[i])**kappa)
   theta2m[i] = T2m[i]*((p0/pS[i])**kappa)

    
if (plotSurface):
   fig = plt.figure()
   cs = plt.plot(t,thetaS,'r-')
   cxl = plt.xlabel('t (hr)')
   cyl = plt.ylabel('theta_s (K)')

   fig = plt.figure()
   cs = plt.plot(t,theta2m,'r-')
   cxl = plt.xlabel('t (hr)')
   cyl = plt.ylabel('theta_2m (K)')


          
          
if (writeSurfaceValues):   
   # Skin real temperature       
   fid = open('surfaceSkinTemperatureTable','w')
   for n in range(nt):
      fid.write('             (' + str(t[n]*3600) + ' ' + str(TS[n]) + ')\n')   
   fid.close()
   
   # 2-m real temperature
   fid = open('surface2mTemperatureTable','w')
   for n in range(nt):
      fid.write('             (' + str(t[n]*3600) + ' ' + str(T2m[n]) + ')\n')   
   fid.close()  
   
   # Skin potential temperature       
   fid = open('surfaceSkinPotentialTemperatureTable','w')
   for n in range(nt):
      fid.write('             (' + str(t[n]*3600) + ' ' + str(thetaS[n]) + ')\n')   
   fid.close()
   
   # 2-m potential temperature
   fid = open('surface2mPotentialTemperatureTable','w')
   for n in range(nt):
      fid.write('             (' + str(t[n]*3600) + ' ' + str(theta2m[n]) + ')\n')   
   fid.close()
   
   # Surface temperature flux
   fid = open('surfaceTemperatureFluxTable','w')
   for n in range(nt):
      fid.write('             (' + str(t[n]*3600) + ' ' + str(-qS[n]) + ')\n')   
   fid.close()
          
          
          
       
       




def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print ("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print ('\t\t%s:' % ncattr,\
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print ("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print ("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print ('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print ("NetCDF dimension information:")
        for dim in nc_dims:
            print ("\tName:", dim) 
            print ("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print ("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print ('\tName:', var)
                print ("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print ("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return (nc_attrs, nc_dims, nc_vars)
