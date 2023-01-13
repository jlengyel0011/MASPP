#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:57:24 2023

@author: sroux
"""


#import seaborn as sns
import warnings
import types
#import numpy.ma as ma
from scipy.stats import t

import numpy as np
import scipy.signal as scsig

import matplotlib.pyplot as plt

# knn libs
from sklearn.neighbors import KDTree
from sklearn.neighbors import NearestNeighbors
from functools import partial

import sys

import pyfftw
import multiprocessing
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon


#sys.path.append("/Users/sroux/CURRENT/TOOLBOXES/GWMFA/MFsynthesis/python/")
#from MFsynthesis20mai2021 import synthmrw
from concurrent.futures import ThreadPoolExecutor


#%% --------------- fftw  functions
pyfftw.interfaces.cache.enable()
#N_threads = multiprocessing.cpu_count()
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
N_threads = multiprocessing.cpu_count()

#%
def prepare_plan2d(Nx, Ny, N_threads = N_threads):
    """
    create plan for fftw
    
    a,fft_a, fft_object, b,ffti_b, ffti_object = prepare_plan2d(Nx, Ny, N_threads = N_threads)

    Parameters
    ----------
    Nx : integer
        x dimension of the plan.
    Ny : integer
        y dimension of the plan
    N_threads : integer, optional
        Number of thread to use. The default is multiprocessing.cpu_count.

    Returns
    -------
    a : numpy array of float64
        aligned memory.
    fft_a : numpy array of complex128
        aligned memory for fft.
    fft_object : fftw object.
        FORWARD fft_object.
    b : numpy array of float64
        aligned memory.
    ffti_b : numpy array of complex128
        aligned memory for fft.
    ffti_object : fftw object.
        BACKWARD' fft_object.

    """
    a     = pyfftw.empty_aligned((Nx, Ny), dtype='float64')
    fft_a = pyfftw.empty_aligned((Nx, Ny//2+1), dtype='complex128')
    fft_object = pyfftw.FFTW(a, fft_a, axes=(0,1), direction='FFTW_FORWARD', threads=N_threads)
        
    ffti_b = pyfftw.empty_aligned((Nx, Ny//2+1), dtype='complex128')
    b = pyfftw.empty_aligned((Nx, Ny), dtype='float64')
    ffti_object = pyfftw.FFTW(ffti_b, b, axes=(0,1), direction='FFTW_BACKWARD', threads=N_threads)
    return a,fft_a, fft_object, b,ffti_b, ffti_object

#%
def do_fftplan2d( N, scales, N_threads = N_threads, wavelet = 'poor'):
    """
    Create fftw plan for fast wavelet transform
    
    a,fft_a, fft_object, b,ffti_b, ffti_object = do_fftplan2d( N, scales, N_threads = N_threads, wavelet = 'poor')

    Parameters
    ----------
    N : integer
        DESCRIPTION.
    scales : float
        vector of positive float.
    N_threads : integer, optional
        Number of thread to use. The default is multiprocessing.cpu_count..
    wavelet : TYPE, optional
        Wavelet to use. The default is 'poor'.

    Returns
    -------
    a : numpy array of float64
        aligned memory.
    fft_a : numpy array of complex128
        aligned memory for fft.
    fft_object : fftw object.
        FORWARD fft_object.
    b : numpy array of float64
        aligned memory.
    ffti_b : numpy array of complex128
        aligned memory for fft.
    ffti_object : fftw object.
        BACKWARD fft_object.

    """
    
    N1, N2 = N
    scales = np.atleast_1d(scales)
    if len(scales) == 1:
        maxradius = int(np.ceil(np.max(scales)))
    else:
        if np.char.equal(wavelet,'poor'):
            maxradius = int(np.ceil(np.max(scales)))
        else:
            maxradius = int(np.ceil(np.sqrt(2)*np.max(scales)))
    
    # add border
    N1 = N1 + 2 * maxradius
    N2 = N2 + 2 * maxradius
    Nk = 2 * maxradius + 1
    
    Nfft1 = N1 + Nk - 1
    Nfft2 = N2 + Nk - 1
    a,fft_a, fft_object, b,ffti_b, ffti_object = prepare_plan2d(Nfft1, Nfft2, N_threads)

    return a,fft_a, fft_object, b,ffti_b, ffti_object

#% 
def convFFTW(a, TFW, fft_object,ffti_object):
    """
    compute the inverse Fourier transform of the product of  
    FFT(a), the Fourier tranform of a,  by TFW. 
    
    temp2 = convFFTW(a, TFW, fft_object,ffti_object)
    
    Parameters
    ----------
    a : numpy array of float64
        signal or image.
    TFW : numpy array of complex128
        Fourier transform of a kernel.
    fft_object : fftw object.
        fftw object to use for FORWARD direction.
    ffti_object : ftw object.
        fftw object to use for BACKWARD direction.

    Returns
    -------
    temp2 : float64
        results.

    """
    b = fft_object(a)
    
    ar=ffti_object(b*TFW)
    temp2=np.copy(ar)
    return temp2
#%% --------------- Usefull functions
#% check input data
def checkinputdata(data):
    """
    Check the type of data.
    Image (uniformly sampled)?
    Marked?
    How many marks?
    
    isimage, ismarked, Nmark = checkinputdata(data)

    Parameters
    ----------
    data : numpy array of float
        The data to check.

    Returns
    -------
    isimage : Boolean
        Equal to 1 if the data is an image.
    ismarked : Boolean
        Equal to 1 if the data ia a marked point process.
    Nmark : int
        Number oif mark in the data.

    """
    # check input data
    si=data.shape
    if len(si)==1:
        print('Error : the input argument must have at least two dimensions')
        return
    else:
        
        if (si[0]>100) & (si[1]>100):
            isimage=1
            if len(si)>2:
                #print('The input seems to be a set of  images  of size ({:d},{:d}))'.format(si[2]),si[0])
                print('The input seems to be a set of  {:d}  images  of size ({:d},{:d})).'.format(si[2],si[0],si[1]))
                ismarked = 1
                Nmark = si[2]
            else:
                print('The input seems to be an image of size ({:d},{:})'.format(si[0],si[1]))
                ismarked = 0
                Nmark = 0
        else:
            isimage=0
            if si[1]>2:
                print('The input seems to be a marked point process with {:} points and {:d} marks.'.format(si[0],si[1]-2))
                
                ismarked = 1
                Nmark = si[1]-2
            else:
                print('The input seems to be a marked point process with {:} points.'.format(si[0]))
                ismarked = 0
                Nmark = 0
    return isimage, ismarked, Nmark

#% optimize block content for non uniformly sampled data
def getoptimblock(datapos, Nanalyse, thresh = 200):
    """
    Define best block positions and sizes.
    
    centerstot, sizeblocktot = getoptimblock(datapos, Nanalyse, thresh = 20)
    

    Parameters
    ----------
    datapos : numpy array of float
        list of x and Y position of N points. 
        Array of shape (N, 2).
    Nanalyse : integer
        Number of points used in the batch.
        Must be set accordingly to the memory available.
        
    thresh : float, optional
        Divide block until the number of point inside pass below this threshold.
        The default is 20.

    Returns
    -------
    centerstot : numpy array of float
        List of x and y position of the center of the blocks = shape (Nblock,2).
    sizeblocktot : numpy array of float
        List of size of the blocks = shape (Nblock,).

    """
    
    sizetmp = Nanalyse
    temp = datapos // sizetmp
    block, count = np.unique(temp, axis=0, return_counts = True)
    sizeblock = count * 0 + sizetmp
    centers = block * sizetmp + sizetmp//2   
    #Nblock = len(block)
    
    # above the threshod : we continue to divide
    index, = np.where(count > 2**thresh)
    sizetmp2 = sizetmp // 2
    #blocktot = block[0:index[0],:]
    if len(index)>0:
        counttot  = count[0:index[0]]
        centerstot = centers[0:index[0],:]
        sizeblocktot = sizeblock[0:index[0]]
        
        for iind in range(len(index)):
            iblock = index[iind]
            #t0=timer()
            ii = np.argwhere( np.all((temp - block[iblock]) == 0, axis = 1))
            #print(timer()-t0)
            temp2 = np.squeeze(np.copy(datapos[ii,:]))
            mitemp2 = np.min(temp2,axis = 0)
            temp2 = temp2 - mitemp2[:,np.newaxis].T
            #matmp=np.ceil(np.max(data,axis=0)).astype(int)
            temp2=temp2 // (sizetmp2)
            blocktmp2, counttmp2 = np.unique( temp2, axis = 0, return_counts = True)
            #u, indices = np.unique(temp, return_inverse=True)
            centers2 = blocktmp2 * sizetmp2 + sizetmp2 // 2+ mitemp2
            sizeblock2 = counttmp2 * 0 + sizetmp2
            centerstot = np.append(centerstot,centers2, axis = 0)
            sizeblocktot = np.append(sizeblocktot,sizeblock2, axis = 0)
            counttot = np.append(counttot,counttmp2, axis = 0) 
            #
            if iind + 1 < len(index):
                centerstot = np.append(centerstot, centers[iblock+1:index[iind+1],:], axis = 0)
                sizeblocktot = np.append(sizeblocktot, sizeblock[iblock+1:index[iind+1]], axis = 0)
                counttot = np.append(counttot, count[iblock+1:index[iind+1]], axis = 0)
                
            
        centerstot = np.append( centerstot, centers[index[-1]+1:,:], axis = 0)
        sizeblocktot = np.append( sizeblocktot, sizeblock[index[-1]+1:], axis = 0)
        counttot = np.append( counttot, count[index[-1]+1:], axis = 0)
        #print(len(sizeblocktot),centerstot.shape,counttot.shape)
    else:
        centerstot = centers
        sizeblocktot = sizeblock
        
    return centerstot, sizeblocktot
#
def bisquarekernel(x):
    '''
        return bisaquare kernel weights (1-x**2)**2 ewith -1< x< 1
        
        
        w=isquarekernel(x)
        
        Input :
            x vector of value taken between -1 and 1.
            
        Output :            
        
            w weights values
        
     ##
      P.Thiraux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''

    A = (1-x**2)**2
    A[np.abs(x) > 1] = 0
    return A

#%%
def threshkernelangle(kernel,thetaTot,theta,dtheta):
    
    if theta+dtheta/2 > np.pi:
        kernel[(thetaTot > theta+dtheta/2-2*np.pi) & (thetaTot < theta-dtheta/2)]=0
    elif theta-dtheta/2 < -np.pi:
        kernel[(thetaTot < theta-dtheta/2+2*np.pi) & (thetaTot > theta+dtheta/2)]=0
    else:
        kernel[thetaTot <= theta-dtheta/2]=0
        kernel[thetaTot > theta+dtheta/2]=0    
    return kernel
#%%
def fitwithNan(Xvector, Yarray, borne=[]):
    '''
        fit a matrix of dim 2 or 3 along first dimension  
        Take care on infinite and nan values in the Yarray
        
        slope, intercept, (pval,rsquare,rmse, stderr) = fitwithNan(Xvector,Yarray,borne=[])
        
        Inputs :
            
            Xvector : vector of x value
        
            Yarray : array od dim 2 or 3 with the size of th first dimension 
                     corresponding to the size of Xvector
                
            borne=[j1, j2] : index of the interval used for then fit  
                           0 < j1 < j2 < len(Xvector). Default j1=0 and j2=len(Xvector).
                         
                          
       Outputs :
           
           slope : the slopes
           intercept : the interceps
           pval : pvalues
           rsquare :  R^2
           rmse     : root mean sqaure errors 
           stderr    : 


     ##
      S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr

    '''

    # take care of the interval
    if len(borne) == 0:
        j1 = 0
        j2 = Xvector.shape[0]

    else:
        j1 = max(0, borne[0])
        j2 = min(Xvector.shape[0], borne[1])


    # take care of the input
    if Yarray.ndim == 2:
        # reshape Xvector to Yarray
        Xarraytmp = np.tile(Xvector[:, np.newaxis], (1, Yarray.shape[1]))
        # take care of  nan
        ind = np.isfinite(Yarray)
        Xarray = Xarraytmp*np.nan
        Xarray[ind] = Xarraytmp[ind]
        # select scales
        Xarray = Xarray[j1:j2, :]
        Yarray = Yarray[j1:j2, :]
        lasum = sum(ind[j1:j2, :])

    elif Yarray.ndim == 3:
        # reshape Xvector to Yarray
        Xarraytmp = np.tile(
            Xvector[:, np.newaxis, np.newaxis], (1, Yarray.shape[1], Yarray.shape[2]))
        # take care of  nan
        ind = np.isfinite(Yarray)
        Xarray = Xarraytmp*np.nan
        Xarray[ind] = Xarraytmp[ind]
        # select scales
        Xarray = Xarray[j1:j2, :, :]
        Yarray = Yarray[j1:j2, :, :]
        lasum = sum(ind[j1:j2, :, :])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)# statistics
        x_mean = np.nanmean(Xarray, axis=0)
        y_mean = np.nanmean(Yarray, axis=0)
        x_std = np.nanstd(Xarray, axis=0)
        y_std = np.nanstd(Yarray, axis=0)
    
        # Compute co-variance between time series of x_array and y_array over each (lon,lat) grid box.
        cov = np.nansum((Xarray-x_mean)*(Yarray-y_mean), axis=0)/lasum
        # Compute correlation coefficients between time series of x_array and y_array over each (lon,lat) grid box.
        cor = cov/(x_std*y_std)
        # Compute slope between time series of x_array and y_array over each (lon,lat) grid box.
        slope = cov/(x_std**2)
        # Compute intercept between time series of x_array and y_array over each (lon,lat) grid box.
        intercept = y_mean-x_mean*slope
    
        # Compute tstats, stderr, and p_val between time series of x_array and y_array over each (lon,lat) grid box.
        n = lasum
        tstats = cor*np.sqrt(n-2)/np.sqrt(1-cor**2)
        stderr = slope/tstats
    
        p_val = t.sf(tstats, n-2)*2
        # Compute r_square and rmse between time series of x_array and y_array over each (lon,lat) grid box.
        # r_square also equals to cor**2 in 1-variable lineare regression analysis, which can be used for checking.
        r_square = np.nansum((slope*Xarray+intercept-y_mean) **
                             2, axis=0)/np.nansum((Yarray-y_mean)**2, axis=0)
        rmse = np.sqrt(np.nansum((Yarray-slope*Xarray-intercept)**2, axis=0)/n)

    return slope, intercept, (p_val,r_square,rmse,stderr)  

#%%
# %%
def computehistogram(WT,bins):
    """
   
      hist, centers, lastd = computehistogram(WT,bins)
      
      Compute  histogram of the normalized wavelet coefficient
      Return also the standart deviation. 
    
      ##
       S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
       
    """
    if len(WT.shape)==1:
        WT=WT[:,np.newaxis]
        
    Nr=WT.shape[1]
    hist=np.zeros((bins,Nr))
    centers=np.zeros((bins,Nr))
    lastd=np.zeros(Nr,)
    for ir in range(Nr):
        temp=WT[:,ir]
        temp=temp[np.isfinite(temp)]
        
        lastd[ir]=np.std(temp)
        # normalize
        if lastd[ir]>0:
            temp=temp/lastd[ir]

        htmp, bin_edges = np.histogram(temp, bins=bins)
        centers[:,ir]=(bin_edges[:-1]+bin_edges[1:])/2
        dx=np.mean(np.diff(centers[:,ir]))
        hist[:,ir]=htmp/np.sum(htmp)/dx

    return hist, centers, lastd

# %%
def computehistogramNoNorm(WT,bins):
    """
   
      hist, centers, lastd = computehistogram(WT,bins)
      
      Compute  histogram of the normalized wavelet coefficient
      Return also the standart deviation. 
    
      ##
       S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
       
    """
    if len(WT.shape)==1:
        WT=WT[:,np.newaxis]
        Nr=WT.shape[1]
    else:
        Nr=WT.shape[1]
    hist=np.zeros((bins,Nr))
    centers=np.zeros((bins,Nr))
    #lastd=np.zeros(Nr,)
    for ir in range(Nr):
        temp=WT[:,ir]
        temp=temp[np.isfinite(temp)]
        
        #lastd[ir]=np.std(temp)
        # normalize
        #if lastd[ir]>0:
        #    temp=temp/lastd[ir]

        htmp, bin_edges = np.histogram(temp, bins=bins)
        centers[:,ir]=(bin_edges[:-1]+bin_edges[1:])/2
        dx=np.mean(np.diff(centers[:,ir]))
        hist[:,ir]=htmp/np.sum(htmp)/dx

    return hist, centers #, lastd

#
###################################### NEW STUFF
# %%
def restoGeoPandaFrame(gridpoints, radius ,results, crs = "EPSG:2154"):
    """
     gdf_results = restoGeoPandaFrame(gridpoints, radius ,results)

         return a geopanda dataframe

     input :

         gridpoints - two dimensional array with the grid points position [x,y]
         radius     - one dimensional array with scales (>0)
         results    - two dimensional array of size equal len(gridpoints) X len(radius)

    output :

        out - geopanda dataframe

    ##
     S.G.  Roux, ENS Lyon, December 2020, stephane.roux@ens-lyon.fr
     J.L June 2021

    """

    #  grid dataframe
    df_grid = pd.DataFrame({'x':gridpoints[:,0], 'y':gridpoints[:,1]})
    # get all scales in a single dataframe
    j=0
    mystr = 'R'+radius[j].astype(int).astype('str')
    df_data = pd.DataFrame(results[:,j], columns = [mystr])
    for j in range(1,len(radius)):
        mystr = 'R'+radius[j].astype(int).astype('str')
        df_data.loc[:,mystr] = pd.Series(results[:,j], index=df_data.index)

    gridsize = np.abs(df_grid['x'][0] - df_grid['x'][1])
    gdf_results = gpd.GeoDataFrame( df_data, geometry=[Polygon([(x-gridsize/2, y+gridsize/2), (x+gridsize/2, y+gridsize/2), (x+gridsize/2, y-gridsize/2), (x-gridsize/2, y-gridsize/2), (x-gridsize/2, y+gridsize/2)])
                              for x,y in zip(df_grid.x,df_grid.y)])

    gdf_results.crs = crs


    return gdf_results

#
# %%  geo weighting
def EpanechnikovWindow(z):
    return 0.75*(1-z**2) 

def triangularWindow(z):
    return (1 - np.abs(z)) 

def tricubeWindow(z):
    return (70/81)*(1-np.abs(z)**3)**3

def bisquareWindow(z):
    return (1-z**2)**2 

def flatWindow(z):
    return np.ones(z.shape)
#%
def geographicalWeight(dd,T,func):
    """
    
    W = geographicalWeight(dist,T,Nr)
    
    return the weight according to the distance dist 
    using a local environment of T.
    
    Input :

       dist - distance of the point. dist and index as the shame shape
       T    - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       Nr   - number of replication of the weight
        
     Output :

       W - the weight. Array of size (len(dd),Nr)
   
    
    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
     S.G.  Roux, ENS Lyon, November 2021
    """  

    # Weighting function
    z = dd/T #this new
    #W = (1 - np.abs(z)) ## triangular
    #W = 0.75*(1-z**2) ## Epanechnikov
    #W = (70/81)*(1-np.abs(z)**3)**3## Tricube
    #W = (T**2-dd**2)**2/T**4 ## bisquare (original)
    W=func(z)
    W[dd>=T]=0
    W=W/np.sum(W,0) # normalization

    #Wfinal=np.tile(W, (Nr, 1)).T
    
    W=W[:,np.newaxis]
    #print(Wfinal.shape,W2.shape)

    return W

#% averaging functions
def localdist(dist,Tloc):
    distout=np.copy(dist)
    distout[dist>Tloc] = np.nan
    distout[dist<=Tloc] = 1
    return distout

#%%
def logaverage(x,w):
    ##########################
    x[x==0] = np.nan
    #w[w==0] = np.nan
    ##########################
    logx=np.log(np.abs(x))
    logx[np.abs(x)==0]=np.nan
    # weight need to be normalized : done
    average=np.nansum(logx*w,axis=0)
    return average

def logaverage2(x,average,w):
    ##########################
    x[x==0] = np.nan
    #w[w==0] = np.nan
    ##########################
    logx=(np.log(np.abs(x))-average)**2
    logx[np.abs(x)==0]=np.nan
    # weight need to be normalized : done
    average2=np.nansum(logx*w,axis=0)
    
    return average2

# %% ---------function for multithreated computations ------ 
def fill(Idx, Dist, val):
    """
    

    Parameters
    ----------
    Idx : TYPE
        DESCRIPTION.
    Dist : TYPE
        DESCRIPTION.
    val : TYPE
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.
    B : TYPE
        DESCRIPTION.

    """
    N = len(Idx)
    M = max( map( len, Idx))
    A = np.full( (N, M), np.nan)
    B = np.full( (N, M), np.nan)
                
    #print(A.shape)
    for i, (aa, bb) in enumerate( zip( Idx, Dist)):
        A[i, :len(aa)] = val[aa]
        B[i, :len(bb)] = bb
    return A, B
#
def threaded_Count_Valued(In, radiustot, val):
    """
    

    Parameters
    ----------
    In : TYPE
        DESCRIPTION.
    radiustot : TYPE
        DESCRIPTION.
    val : TYPE
        DESCRIPTION.

    Returns
    -------
    Count : TYPE
        DESCRIPTION.

    """
    Idx, Dist = In
    valbis, dd = fill( Idx, Dist, val)
    Count = np.zeros( (valbis.shape[0],len(radiustot),3))
    for ir in range( len( radiustot)):
        
        valbis[dd >= radiustot[-1 - ir]] = np.nan
        Count[:, -1 -ir, 0] = np.nansum( np.isfinite(valbis), axis = 1)
        Count[:, -1 -ir, 1] = np.nanmean( valbis, axis = 1)
        Count[:, -1 -ir, 2] = np.nanstd( valbis, axis = 1)
    
    return Count

#%%
#
def threaded_Log(In, radiustot, val):
    """
    

    Parameters
    ----------
    In : TYPE
        DESCRIPTION.
    radiustot : TYPE
        DESCRIPTION.
    val : TYPE
        DESCRIPTION.

    Returns
    -------
    Count : TYPE
        DESCRIPTION.

    """
    Idx, Dist = In
    Count = np.zeros((len(Idx),len(radiustot)))
    for ir in range(len(radiustot)):
        valbis, dd = fill(Idx,Dist,val[:,ir])
        valbis[dd>=radiustot[-1-ir]]=np.nan
        
        Count[:,-1-ir] = np.nanmean(valbis,axis=1)
    
    return Count

#
def threaded_Log_OneScale(In, radiustot, val):
    Idx, Dist = In
    Count = np.zeros((len(Idx),))
    
    valbis, dd = fill( Idx, Dist, val)
    valbis[dd >= radiustot] = np.nan
    Count = np.nanmean(valbis, axis = 1)
    
    return Count

# %% count point with dist<radius
def threaded_Count(dist , radius):   
    """
    
      out = threaded_Count(dist,radius)
      
      Compute  the number of element with distance lower than radius
      Radius can be a vector.
    
      ##
       S.G.  Roux, ENS Lyon, November 2021,  stephane.roux@ens-lyon.fr
       
    """    

    radius = np.atleast_1d(radius)
    Count = np.zeros( ( 1, radius.shape[0]), dtype = 'float')
    for i in range(radius.size):
        dist = dist[ dist< radius[ -1 - i]]
        Count[ 0,-1 - i] = dist.shape[0]
        
    return Count

#%% function for the local cumulant (log of the coefs)
def locallogCoefs(data, Wcoefs, radius, Nanalyse = 2**14, NonUniformData = False, verbose = True):
    """
    

    Parameters
    ----------
    Wcoef : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    Nanalyse : TYPE, optional
        DESCRIPTION. The default is 2**14.
    NonUniformData : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    int
        DESCRIPTION.

    """
    
    #%% check input data
    si = data.shape
    if si[1] != 2 : 
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')
     
    # select the destination set of points
    destination = data[ :, 0:2]
    N = destination.shape[0]
    
    # check the analysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        lcoefs  = np.log(np.abs(Wcoefs))
    Nscales = Wcoefs.shape[1]
    Npts = Wcoefs.shape[0]
    if Npts != N:
        raise TypeError("not a good number of points")
   
    radius = np.atleast_1d(radius)
    radiustot = np.sort(radius)
    temp, temp2, indexreadius = np.intersect1d( radius, radiustot, return_indices = True)
    if radius.shape[0] != Nscales:
        raise TypeError("not a good number of scales")
    # some constants
    scalemax = np.max(radiustot)
    
    
    # sub parameters to partition  the data
    if Nanalyse == 0: # this is dangerous
        print('Warning : dangerous if your set of data is large.')
        Nanalyse = N
        Nbunch = 1
    else:
        # find how many bunchs
        if NonUniformData:
            #t1=timer()
            centers, sizeblock = getoptimblock( data, Nanalyse)
            #t2=timer()
            if verbose:
                print('Block optimisation for non uniformly spaced data.')
            Nbunch = len(sizeblock)
        else:
            #t1=timer()
            sizetmp = Nanalyse
            temp=data // sizetmp
            block, count = np.unique( temp, axis = 0, return_counts = True)
            centers = block*sizetmp + sizetmp // 2
            sizeblock = count * 0 + sizetmp
            Nbunch = len(block)
            #t2=timer()
            if verbose:
                print('No block optimisation uniformly spaced data.')
            
    
    Nanalyse2 = 2**22
    #print('Nanalyse (bunch) and Nanalyse2 (block)',  Nanalyse, Nanalyse2,t2-t1)
    
    if verbose:
        print('Computation in {:d} bunches :'.format(Nbunch), end = ' ')
    
    # allocation
    LWCoefs = np.nan*np.zeros((N, radius.shape[0]), dtype = float)

    
    # set the worker for knn search
    neigh = NearestNeighbors( n_jobs = 4)
    
    #%% loop on the bunches
    for ibunch in range( Nbunch):
        #%%
        print(ibunch + 1, end=' ')
        sys.stdout.flush()
              
        center = centers[ ibunch, :]
        sizetmp = sizeblock[ ibunch]
        #grrr
        index,= np.where(((data[:,0] >= center[0]-sizetmp//2) & (data[:,0]< center[0]+sizetmp//2)) & ((data[:,1]>= center[1]-sizetmp//2) & (data[:,1]< center[1]+sizetmp//2)))
        IndexNear, = np.where(((data[:,0] >= center[0]-(sizetmp//2+scalemax+1)) & (data[:,0]< center[0]+(sizetmp//2+scalemax+1))) & ((data[:,1]>= center[1]-(sizetmp//2+scalemax+1)) & (data[:,1]< center[1]+(sizetmp//2+scalemax+1))))
        neigh.fit( data[ IndexNear, :])
        
        # search neiarest neighbors
        Disttot, IdxTot = neigh.radius_neighbors(destination[index,0:2], radius = scalemax, return_distance = True)
        # get the max length of neighbors --> computation by block
        Maxlength = max(map(len, IdxTot))
        Nblock = int(np.ceil( index.shape[0] * Maxlength / Nanalyse2))
        Nptx = len(index) // Nblock+1
        # correct the number of block if needed
        if (Nblock -1) * Nptx > min( len(index), Nblock*Nptx):
            Nblock = Nblock - 1
        
        isallscales = False
        if isallscales:
            # All scales together
            malist2 = [(IdxTot[i*Nptx:min(len(index),(i+1)*Nptx)], Disttot[i*Nptx:min(len(index),(i+1)*Nptx)]) for i in range(Nblock)]
            partialknnMeanLog = partial( threaded_Log, radiustot = radiustot, val = lcoefs[ IndexNear, :])
            with ThreadPoolExecutor() as executor:
                result_list3 = executor.map( partialknnMeanLog, malist2)
            LWCoefs[ index, :] = np.vstack( list( result_list3))
            
        else:
            for ir in range( Nscales):
                radius = radiustot[ ir]
                malist2 = [(IdxTot[i*Nptx:min(len(index),(i+1)*Nptx)], Disttot[i*Nptx:min(len(index),(i+1)*Nptx)]) for i in range(Nblock)]
               
                partialknnMeanLog = partial( threaded_Log_OneScale, radiustot = radius, val = lcoefs[IndexNear,ir])
                with ThreadPoolExecutor() as executor:
                    result_list4 = executor.map( partialknnMeanLog, malist2)
                
                LWCoefs[ index, ir] = np.hstack( list( result_list4))

        
        
    print('.')    
    return LWCoefs

#%%
def localWaveTrans(data, radius, Nanalyse = 2**14, destination = np.array([]), NonUniformData = False, verbose = True):
    """
    
     [Count, Wave, CountG] = aveTrans(source, radius, T=None, Nanalyse=2**16, destination = []))
    
    
    
    Compute box-counting and wavelet coefficient on a valued/non valued set of data points.
    
    If the data are not valued, count for every data point
            -- the number of neighboors in ball of radius r (GWFA) : N(r).
            -- the wavelet coeficient at scale r (GWMFA) : 2*N(r)-N(sqrt(2)*r).
    
    If the data are valued, count for every datapoint
            -- the number of neighboors in ball of radius r, the mean and std of the marked value
            -- the wavelet coeficient at scale r on the marked value.
    
    Input :
    
        source     - Non-marked  point process : matrix of size N X 2 for N points
                        where source[i,:]=[X_i,Y_i] with 2D cooprdonate of point i.
                      Marked  point process :  matrix of size Nx3
                        where source[i,:]=[X_i,Y_i, mark_i] with 2D cooprdonate of point i and value
        radius      - list of scales to be investigated
        Nanalyse    - number of points to analyse in one bach. Default is 2**16 points.
                        If Nanalyse=0, compute all the points in once (dangerous!!!)
        destination - Non marked point process (destination_i=[X_I,Y_i]) where the coeficient are calculated
                        Default empty : compute at source position.
    Output :
    
        Count    - matrix of size Nxlength(radius) with box-counting coefficients
        Wave     - matrix of size Nxlength(radius) with wavelet coefficients
        Count    - matrix of size Nx2 with box-counting and wavelet coeficient at scale T
                  
    
    
    Usage exemple :
    
    
     ##
      S.G.  Roux, ENS Lyon, November 2021,  stephane.roux@ens-lyon.fr
      J.L. June 2021
    """
 
    #%% check input data
    si = data.shape
    if si[1] < 2: # or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')
     
    # select the destination set of points
    if destination.shape[0] == 0: 
        destination = data[:,0:2]
    else:
        sid = destination.shape
        #print(sid)
        if sid[1] != 2 :
            raise TypeError('The ''destination''  argument must be of length 2.')
            
    # check the analysis
    isvalued = 0 
    if si[1] > 2: # valued analysis
        isvalued = 1
        if verbose:
            print('Valued data analysis on {:d} points to {:d} destination points.'.format(data.shape[0],destination.shape[0]))
        val = data[:,2]
        #valtot = data[:,2:]
        data = data[:,0:2]
        
    else:           # non valued analysis
        val = data[:,0] * 0 + 1
        if verbose:
            print('Non valued data analysis on {:d} points to {:d} destination points.'.format(data.shape[0],destination.shape[0]))
    
    radius = np.atleast_1d(radius)
    radiustot = np.sort(radius)
    #temp,temp2,indexreadius=np.intersect1d(radius,radiustot, return_indices=True)

    # some constants
    scalemax=np.max(radiustot)
    N = destination.shape[0]
    
    # sub parameters to partition  the data
    if Nanalyse==0: # this is dangerous
        print('Warning : dangerous if your set of data is large.')
        Nanalyse=N
        Nbunch=1
    else:
        # find how many bunchs
        if NonUniformData:
            centers, sizeblock = getoptimblock(data,Nanalyse)
            Nbunch = len(sizeblock)
            if verbose:
                print('Block optimisation for non uniformly spaced data.')
                       
        else:
            sizetmp = Nanalyse
            temp= data // sizetmp
            block, count = np.unique(temp, axis=0, return_counts=True)
            centers = block*sizetmp+sizetmp//2
            sizeblock = count*0+sizetmp
            Nbunch = len(block)
            if verbose:
                print('No block optimisation uniformly spaced data.')
            
    
    Nanalyse2 = 2**22
    #print('Nanalyse (bunch) and Nanalyse2 (block)',  Nanalyse, Nanalyse2,t2-t1)
    if verbose:
        print('Computation in {:d} bunches :'.format(Nbunch),end=' ')  
    
   
    if isvalued: # we compute mean, std and count
        Count = np.nan*np.zeros((N,radius.shape[0],3),dtype=float)
        #  store count of valid value, mean and std of the point process mark           
    else:         #  we compute only count
        Count = np.nan*np.zeros((N,radius.shape[0]),dtype=float)
    
    # set the worker for knn search
    neigh = NearestNeighbors(n_jobs=4)
    
    #%% loop on the bunches
    for ibunch in range(Nbunch):
        #%%
        print(ibunch+1, end=' ')
        sys.stdout.flush()
        #
        center = centers[ibunch,:]
        sizetmp = sizeblock[ibunch]
        #
        index, = np.where(((data[:,0] >= center[0]-sizetmp//2) & (data[:,0]< center[0]+sizetmp//2)) & ((data[:,1]>= center[1]-sizetmp//2) & (data[:,1]< center[1]+sizetmp//2)))
        IndexNear, = np.where(((data[:,0] >= center[0]-(sizetmp//2+scalemax+1)) & (data[:,0]< center[0]+(sizetmp//2+scalemax+1))) & ((data[:,1]>= center[1]-(sizetmp//2+scalemax+1)) & (data[:,1]< center[1]+(sizetmp//2+scalemax+1))))
        neigh.fit(data[IndexNear,:])
        
        if isvalued:  # valued analysis
            # search neiarest neighbors
            #t1=timer();
            Disttot, IdxTot = neigh.radius_neighbors(destination[index,0:2], radius = scalemax, return_distance = True)
            #t2=timer();
            #print('knn search',t2-t1)
            # get the max length of neighbors --> computation by block
            #t1=timer();
            #print(count[ibunch],len(Disttot))
            Maxlength = max(map( len, IdxTot))
            #print('Npts/max Nneighbor ',count[ibunch],Maxlength)
            
            Nblock = int(np.ceil(index.shape[0] * Maxlength / Nanalyse2))           
            Nptx = len(index) // Nblock + 1
            #print(Nblock,Nptx)
            # correct the number of block if needed
            if (Nblock - 1)*Nptx > min(len(index), Nblock * Nptx):
                Nblock = Nblock - 1
            
            #print(Nblock,Nptx)
            malist2=[(IdxTot[i*Nptx:min(len(index),(i+1)*Nptx)], Disttot[i*Nptx:min(len(index),(i+1)*Nptx)]) for i in range(Nblock)]
            partialknncountbis = partial(threaded_Count_Valued, radiustot = radiustot,val = val[IndexNear])
            with ThreadPoolExecutor() as executor:
                result_list3 = executor.map(partialknncountbis, malist2)
            
            Count[index,:,:] = np.vstack(list(result_list3))
            #t2=timer();
            #print('count2 ',t2-t1)
            
        else:  # non valued analysis
            # change to neiarest library for workers
            Disttot, IdxTot = neigh.radius_neighbors(destination[index,0:2],radius=scalemax,return_distance=True)
            partialcount = partial(threaded_Count, radius=radiustot)
            with ThreadPoolExecutor() as executor:
                result_list = executor.map(partialcount, Disttot)
            
            #print(len(list(result_list)),)
            temp = np.squeeze(np.array(list(result_list)));
            #print(temp.shape,Count[index,:].shape)
            if radius.shape[0] == 1:
                Count[index,:] = temp[:,np.newaxis]
            else:    
                Count[index,:] = temp
            
    del neigh
    print('.')
    
    if isvalued: # compute poor coefs       
        Wcoef = np.copy(Count)
        Wcoef[:,:,1] = Wcoef[:,:,1] - val[:,np.newaxis]    
        return Wcoef, Count
    
    else:
        return Count
#%% --------------- 1D Convolution 
#
def NormWithNaNConv1d(narray, kernel):
    '''
     ##
      P.Thiraux & S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''

    narray[np.isinf(narray)] = np.nan
    ind1 = np.isfinite(narray)
    tmp1 = np.zeros(narray.shape)
    tmp1[ind1] = 1
    
    NewNorm =scsig.fftconvolve(tmp1, kernel, mode='valid')
    
    
    NewNorm[np.where(NewNorm < 1e-12)[0]] = np.nan
    
    
    return NewNorm

#%%
def ContinuousHarrDecomp1d(data,scales):
    '''
    
     Continuous Wavelet decomposition using Harr wavelet.
     The wavelet is a difference of box or an average of increment (of size scale) 
     over a window of size scale. 
     
     Coefs,Nr = ContinuousHarrDecomp1d(signal,scales)

        Inputs :
            
            signal : signal to analyze (must be one dimensional)
            
            scales : vector of scales values
            
            
        Outputs :            
            
            Coefs : wavelet coeficients for all the scales  using poor wavelet
            
            Nr : number of valid increments in ball of size scales
                 #(x' |  |x-x'| < scale) - signal(x)
            

     ##
      S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''
    
    # size parameter   
    N = len(data)    
    Nscales = len(scales)
    
    # allocations
    Coefs = np.zeros((Nscales,N))*np.nan
    Nr = np.zeros((Nscales,N))*np.nan
    Cumulant = np.zeros((Nscales,N))*np.nan
    # loop on scales
    for ir in range(Nscales):
        print(ir,end=' ',flush=True)
        r = int(scales[ir])
        # 
        W_r = np.ones(r,)
        
        # temporary allocation
        approx = np.zeros(data.shape)*np.nan
        # increments of size r
        approx[r::] = data[r::]-data[:-r:]
        # normalisation for average (taking care of Nan)
        Normsignal1 = NormWithNaNConv1d(approx, W_r)
        # take care of NaN       
        approx[np.isnan(approx)]=0   
        # average
        Coefs[ir,0:N-(r-1)] = scsig.fftconvolve(approx,W_r, mode='valid')/Normsignal1
        Nr[ir,0:N-(r-1)] = Normsignal1
        # log of coef
        LCoefs = np.log(abs(Coefs[ir,:]))
        LCoefs[np.where(Coefs[ir,:]==0)]=np.nan        
        isnan = np.isnan(LCoefs)
        Wlog = np.ones(2*r+1)
        NewNormL1_r = NormWithNaNConv1d(LCoefs,Wlog)
                
        LCoefs[isnan]=0
        Cumulant[ir,r:N-r] = scsig.fftconvolve(LCoefs,Wlog,mode='valid')/NewNormL1_r
        Cumulant[ir,isnan]=np.nan
        
    print('.')
           
    return Coefs, Nr, Cumulant
#
def computeHarrCoefsConv1d(signal, echelle):
    """
    
        Compute wavelet coefficients for differents scales of a signal regulary sampled
        The signal can contain NaN values.
        The following wavelets are used : 
            poor wavelet : t(x,r)= mean (x' |  |x-x'| < scale) - signal(x)
            hat wavelet  : t(x,r)=  mean (x' |  |x-x'| < scale) - mean( x' |  scale < |x-x'| < 2*scale)
        
        
        Coefs, CoefsHat, Nr, dNr, lcoefs, lcoefsHat = computeCoefsConv1d(signal, scales):

        Inputs :
            
            signal : signal to analyze (must be one dimensional)
            
            scales : vector of scales values
            
            
        Outputs :            
            
            Coefs : wavelet coeficients for all the scales  using poor wavelet
            
            CoefsHat : wavelet coeficients for all the scales  using hat wavelet
            
            Nr : number of valid values in ball of size scales
                 #(x' |  |x-x'| < scale) - signal(x)
            
            dNr : differnce of number of valid value 
                  #(x' |  |x-x'| < scale) - #( x' |  scale < |x-x'| < 2*scale)
                  
            lcoefs : average eof the log of the wavelet coeficients over a ball of size scale.
                     The poor wavelet is used.
                     
            lcoefsHat : same as lcoefs but using hat wavelet
              
     ##
      P.Thiraux & S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    """
    
    #N0= len(signal)
    
    signal1 = np.squeeze(signal)
    #IndNan = np.isnan(signal1)
    # padding for no border effect
    Nborder=np.max(echelle)
    signal1=np.pad(signal1,(Nborder, Nborder), constant_values=(np.nan, np.nan))
    IndNan = np.isnan(signal1)
    sig1copy = np.copy(signal1)
    sig1copy[np.isnan(signal1)] = 0

    N = len(signal1)  # Les signaux sont de mêmes tailles

    Nr=np.zeros((len(echelle), len(signal1)))*np.nan
    #Nrb=np.zeros((len(echelle), len(signal1)))*np.nan
    Nr2=np.zeros((len(echelle), len(signal1)))*np.nan
    Nr_bis=np.zeros((len(echelle), len(signal1)))*np.nan
    Mean = np.zeros((len(echelle), len(signal1)))*np.nan
    LMean = np.zeros((len(echelle), len(signal1)))*np.nan
    
    Std = np.zeros((len(echelle), len(signal1)))*np.nan
    Std2 = np.zeros((len(echelle), len(signal1)))*np.nan
    Poor = np.zeros((len(echelle), len(signal1)))*np.nan
    HatMean = np.zeros((len(echelle), len(signal1)))*np.nan
    Coefs22 = np.zeros((len(echelle), len(signal1)))*np.nan
    LPoor = np.zeros((len(echelle), len(signal1)))*np.nan
    #Cumulant2 = np.zeros((len(echelle), len(signal1)))*np.nan
    LHatMean = np.zeros((len(echelle), len(signal1)))*np.nan
    
    for i in range(len(echelle)):
        print(i,end=' ',flush=True)
        r = echelle[i]
        Coefstmp = np.zeros(N)*np.nan
        
        """Calcul des Coefficients multi-échelle"""
        W_r = np.ones(2*r+1)
        W_r [0:r] =0
        W_r [-1] =0
        #plt.clf()
        #plt.plot(np.arange(-r,r+1,1),W_r,'k.')
        #print('premier ',np.sum(W_r))
        #W_r[r]=0
        Normsignal1 = NormWithNaNConv1d(signal1, W_r) 
        Nr[i,r:N - r]=Normsignal1
        #Nr[i,:]=Normsignal1
        Nr[i,np.isnan(signal1)] = np.nan
        #Coefstmp[r:N - r] = np.convolve(sig1copy,W_r, mode='valid')/Normsignal1
        Coefstmp[r:N - r] = scsig.fftconvolve(sig1copy,W_r, mode='valid')/Normsignal1
        #Coefstmp[r:N - r] =toto
        #print(toto.shape,)
        #print('norm  ', np.nanmin((Coefstmp[r:N - r]-toto)))
        #grrr
        Coefstmp[IndNan] =  np.nan
        Mean[i,:]  = Coefstmp
        Poor[i,:]  = Mean[i,:] - signal1
        # Mom2 for std of poor coef (the same as std of the signal)
        Mom2tmp = np.zeros(N)*np.nan
        Mom2tmp[r:N - r] = scsig.fftconvolve(sig1copy**2,W_r, mode='valid')/Normsignal1        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lavar=np.maximum(Mom2tmp-Mean[i,:]**2,0)
            lastd=np.sqrt(lavar)  
        lastd[IndNan] = np.nan
        Std[i,:] = lastd
        
        # log of coef
        LCoefs = np.log(abs(Poor[i,:]))
        LCoefs[np.where(Poor[i,:]==0)]=np.nan        
        isnan = np.isnan(LCoefs)
        Wlog = np.ones(2*r+1)
        NewNormL1_r = NormWithNaNConv1d(LCoefs,Wlog)
        #NewNormL2_r = NormWithNaNConv1d(LCoefs**2,Wlog)
        
        LCoefs[isnan]=0
        LPoor[i,r:N-r] = scsig.fftconvolve(LCoefs,Wlog,mode='valid')/NewNormL1_r
        LPoor[i,isnan]=np.nan
        
        #Cumulant2[i,r:N-r] = scsig.fftconvolve(LCoefs**2,Wlog,mode='valid')/NewNormL2_r
        #Cumulant2[i,r:N-r] = Cumulant2[i,r:N-r] -LPoor[i,r:N-r]**2
        #Cumulant2[i,isnan]=np.nan
        
        # --> hat wavelet
        # first scale
        #Normsignal1b = NormWithNaNConv1d(signal1, W_r)
        #Nrb[i,r:N - r]=Normsignal1b
        #Nrb[i,np.isnan(signal1)] = np.nan
        #Coefstmp[r:N - r] = scsig.fftconvolve(sig1copy,W_r, mode='valid')/Normsignal1b
        # second scale
        #r_bis=(r*2).astype(int)
        #W_r_bis = np.ones(2*r_bis+1)
        W_r_bis = np.ones(2*r+1)
        W_r_bis [r:] =0
        #plt.plot(np.arange(-r,r+1,1),W_r_bis,'r+')
        #print('deuxieme ',np.sum(W_r_bis))
        #W_r_bis[r_bis-r:r_bis+r+1] = 0        
        Normsignal2 = NormWithNaNConv1d(signal1, W_r_bis)
        Nr2[i,r:N - r]=Normsignal2
        Nr2[i,np.isnan(signal1)] = np.nan
        Coefs22[i,r:N - r] = scsig.fftconvolve(sig1copy,W_r_bis, mode='valid')/Normsignal2
        # Mom2 pour Std
        Mom2tmp = np.zeros(N)*np.nan
        Mom2tmp[r:N - r] = scsig.fftconvolve(sig1copy,W_r_bis, mode='valid')/Normsignal2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lavar=np.maximum(Mom2tmp-Coefs22[i,:]**2,0)
            lastd=np.sqrt(lavar)  
        lastd[IndNan] = np.nan
        Std2[i,:] = lastd
        
        HatMean[i,:] = Coefstmp-Coefs22[i,:]
        HatMean[i,np.isnan(signal1)] = np.nan
        Nr_bis[i,:] = 2*Nr[i,:] - Nr2[i,:]      
        
        # log of coef
        LCoefs = np.log(abs(HatMean[i,:]))
        LCoefs[np.where(HatMean[i,:]==0)]=np.nan        
        isnan = np.isnan(LCoefs)
        Wlog = np.ones(2*r+1)
        NewNormL1_r = NormWithNaNConv1d(LCoefs,Wlog)
        
        LCoefs[isnan]=0
        LHatMean[i,r:N-r] = scsig.fftconvolve(LCoefs,Wlog,mode='valid')/NewNormL1_r
        LHatMean[i,isnan]=np.nan
        
    print('.')
    # go back to the input size
    Mean=Mean[:,np.max(echelle):-np.max(echelle)]
    LMean=LMean[:,np.max(echelle):-np.max(echelle)]
    
    Poor=Poor[:,np.max(echelle):-np.max(echelle)]
    LPoor=LPoor[:,np.max(echelle):-np.max(echelle)]
    HatMean=HatMean[:,np.max(echelle):-np.max(echelle)]
    LHatMean=LHatMean[:,np.max(echelle):-np.max(echelle)]
    
    Nr=Nr[:,np.max(echelle):-np.max(echelle)]
    Nr_bis=Nr_bis[:,np.max(echelle):-np.max(echelle)]
    
    

    HatStd = Std[:,np.max(echelle):-np.max(echelle)]  - Std2[:,np.max(echelle):-np.max(echelle)] 
    Std = Std[:,np.max(echelle):-np.max(echelle)] 
    #toto = Nr, Nr_bis, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd 
    
    return Nr.T, Nr_bis.T, Mean.T, LMean.T, HatMean.T, LHatMean.T, Poor.T, LPoor.T, Std.T, HatStd.T
    #return Coefs, Coefs_bis, Nr, Nr_bis, Cumulant, Cumulant_bis#, Cumulant2

#
#
def computeCoefsConv1d(signal, echelle):
    """
    
        Compute wavelet coefficients for differents scales of a signal regulary sampled
        The signal can contain NaN values.
        The following wavelets are used : 
            poor wavelet : t(x,r)= mean (x' |  |x-x'| < scale) - signal(x)
            hat wavelet  : t(x,r)=  mean (x' |  |x-x'| < scale) - mean( x' |  scale < |x-x'| < 2*scale)
        
        
        Coefs, CoefsHat, Nr, dNr, lcoefs, lcoefsHat = computeCoefsConv1d(signal, scales):

        Inputs :
            
            signal : signal to analyze (must be one dimensional)
            
            scales : vector of scales values
            
            
        Outputs :            
            
            Coefs : wavelet coeficients for all the scales  using poor wavelet
            
            CoefsHat : wavelet coeficients for all the scales  using hat wavelet
            
            Nr : number of valid values in ball of size scales
                 #(x' |  |x-x'| < scale) - signal(x)
            
            dNr : differnce of number of valid value 
                  #(x' |  |x-x'| < scale) - #( x' |  scale < |x-x'| < 2*scale)
                  
            lcoefs : average eof the log of the wavelet coeficients over a ball of size scale.
                     The poor wavelet is used.
                     
            lcoefsHat : same as lcoefs but using hat wavelet
              
     ##
      P.Thiraux & S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    """
    
    #N0= len(signal)
    
    signal1 = np.squeeze(signal)
    #IndNan = np.isnan(signal1)
    # padding for no border effect
    Nborder=np.max(echelle)
    signal1=np.pad(signal1,(Nborder, Nborder), constant_values=(np.nan, np.nan))
    IndNan = np.isnan(signal1)
    sig1copy = np.copy(signal1)
    sig1copy[np.isnan(signal1)] = 0
    
    N = len(signal1)  # Les signaux sont de mêmes tailles
    #print(N, Nborder)
    #gt
    Nr=np.zeros((len(echelle), len(signal1)))*np.nan
    #Nrb=np.zeros((len(echelle), len(signal1)))*np.nan
    Nr2=np.zeros((len(echelle), len(signal1)))*np.nan
    Nr_bis=np.zeros((len(echelle), len(signal1)))*np.nan
    Mean = np.zeros((len(echelle), len(signal1)))*np.nan
    LMean = np.zeros((len(echelle), len(signal1)))*np.nan
    
    Std = np.zeros((len(echelle), len(signal1)))*np.nan
    Std2 = np.zeros((len(echelle), len(signal1)))*np.nan
    Poor = np.zeros((len(echelle), len(signal1)))*np.nan
    HatMean = np.zeros((len(echelle), len(signal1)))*np.nan
    Coefs22 = np.zeros((len(echelle), len(signal1)))*np.nan
    LPoor = np.zeros((len(echelle), len(signal1)))*np.nan
    #Cumulant2 = np.zeros((len(echelle), len(signal1)))*np.nan
    LHatMean = np.zeros((len(echelle), len(signal1)))*np.nan
    
    for i in range(len(echelle)):
        print(i,end=' ',flush=True)
        r = echelle[i]
        Coefstmp = np.zeros(N)*np.nan
        
        """Calcul des Coefficients multi-échelle"""
        W_r = np.ones(2*r+1)
        #W_r[r]=0
        Normsignal1 = NormWithNaNConv1d(signal1, W_r) 
        Nr[i,r:N - r]=Normsignal1
        Nr[i,np.isnan(signal1)] = np.nan
        Coefstmp[r:N - r] = scsig.fftconvolve(sig1copy,W_r, mode='valid')/Normsignal1
        Coefstmp[IndNan] =  np.nan
        Mean[i,:]  = Coefstmp
        Poor[i,:]  = Mean[i,:] - signal1
        # Mom2 for std of poor coef (the same as std of the signal)
        Mom2tmp = np.zeros(N)*np.nan
        Mom2tmp[r:N - r] = scsig.fftconvolve(sig1copy**2,W_r, mode='valid')/Normsignal1        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lavar=np.maximum(Mom2tmp-Mean[i,:]**2,0)
            lastd=np.sqrt(lavar)  
        lastd[IndNan] = np.nan
        Std[i,:] = lastd
        
        # log of coef
        LCoefs = np.log(abs(Poor[i,:]))
        LCoefs[np.where(Poor[i,:]==0)]=np.nan        
        isnan = np.isnan(LCoefs)
        Wlog = np.ones(2*r+1)
        NewNormL1_r = NormWithNaNConv1d(LCoefs,Wlog)
        
        LCoefs[isnan]=0
        LPoor[i,r:N-r] = scsig.fftconvolve(LCoefs,Wlog,mode='valid')/NewNormL1_r
        LPoor[i,isnan]=np.nan
        
        #Cumulant2[i,r:N-r] = scsig.fftconvolve(LCoefs**2,Wlog,mode='valid')/NewNormL2_r
        #Cumulant2[i,r:N-r] = Cumulant2[i,r:N-r] -LPoor[i,r:N-r]**2
        #Cumulant2[i,isnan]=np.nan
        
        # --> hat wavelet
        # first scale
        #Normsignal1b = NormWithNaNConv1d(signal1, W_r)
        #Nrb[i,r:N - r]=Normsignal1b
        #Nrb[i,np.isnan(signal1)] = np.nan
        #Coefstmp[r:N - r] = scsig.fftconvolve(sig1copy,W_r, mode='valid')/Normsignal1b
        # second scale
        r_bis=(r*2).astype(int)
        W_r_bis = np.ones(2*r_bis+1)
        #W_r_bis[r_bis-r:r_bis+r+1] = 0        
        Normsignal2 = NormWithNaNConv1d(signal1, W_r_bis)
        Nr2[i,r_bis:N - r_bis]=Normsignal2
        Nr2[i,np.isnan(signal1)] = np.nan
        Coefs22[i,r_bis:N - r_bis] = scsig.fftconvolve(sig1copy,W_r_bis, mode='valid')/Normsignal2
        # Mom2 pour Std
        Mom2tmp = np.zeros(N)*np.nan
        Mom2tmp[r_bis:N - r_bis] = scsig.fftconvolve(sig1copy,W_r_bis, mode='valid')/Normsignal2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            lavar=np.maximum(Mom2tmp-Coefs22[i,:]**2,0)
            lastd=np.sqrt(lavar)  
        lastd[IndNan] = np.nan
        Std2[i,:] = lastd
        
        HatMean[i,:] = Coefstmp-Coefs22[i,:]
        HatMean[i,np.isnan(signal1)] = np.nan
        Nr_bis[i,:] = 2*Nr[i,:] - Nr2[i,:]      
        
        # log of coef
        LCoefs = np.log(abs(HatMean[i,:]))
        LCoefs[np.where(HatMean[i,:]==0)]=np.nan        
        isnan = np.isnan(LCoefs)
        Wlog = np.ones(2*r+1)
        NewNormL1_r = NormWithNaNConv1d(LCoefs,Wlog)
        
        LCoefs[isnan]=0
        LHatMean[i,r:N-r] = scsig.fftconvolve(LCoefs,Wlog,mode='valid')/NewNormL1_r
        LHatMean[i,isnan]=np.nan
        
    print('.')
    # go back to the input size
    Mean=Mean[:,np.max(echelle):-np.max(echelle)]
    LMean=LMean[:,np.max(echelle):-np.max(echelle)]
    
    Poor=Poor[:,np.max(echelle):-np.max(echelle)]
    LPoor=LPoor[:,np.max(echelle):-np.max(echelle)]
    HatMean=HatMean[:,np.max(echelle):-np.max(echelle)]
    LHatMean=LHatMean[:,np.max(echelle):-np.max(echelle)]
    
    Nr=Nr[:,np.max(echelle):-np.max(echelle)]
    Nr_bis=Nr_bis[:,np.max(echelle):-np.max(echelle)]
    
    
    #plt.plot(Nr[i,:] ,'g+')
    #grrr
    HatStd = Std[:,np.max(echelle):-np.max(echelle)]  - Std2[:,np.max(echelle):-np.max(echelle)] 
    Std = Std[:,np.max(echelle):-np.max(echelle)] 
    #toto = Nr, Nr_bis, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd 
    
    return Nr.T, Nr_bis.T, Mean.T, LMean.T, HatMean.T, LHatMean.T, Poor.T, LPoor.T, Std.T, HatStd.T
    #return Coefs, Coefs_bis, Nr, Nr_bis, Cumulant, Cumulant_bis#, Cumulant2

#
def localMomCumConv1d(coefs,lcoefs,L,window='flat', gridsize=1):
    '''
        Compute localized statistics (moments and cumulants).
        
        Mom1, Mom2, Flatness ,Cum1, Cum2 = localMomCumConv1d(coefs,lcoefs,L,window)
        
        Inputs :
            
            coefs  : set of ceoficients used to compute moments and flatness
            
            lcoefs:  set of log coefficients used to compute the cumulants
            
            L : size of the average (integer stricty positif)
            
            window : weighting function to use. Default : window='flat'.
                     can be window='bisquare'
            
        Outputs :   
            
            Mom1, Mom2 : localized moment of order one and two
            
            Flatness : localized flatness
            
            cum1, cum2 : localized log cumulant of order one and two
            
        
     ##
      P.Thiraux  S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''
    assert(coefs.shape == lcoefs.shape)
    # swap axis
    coefs = coefs.T
    lcoefs = lcoefs.T
    coefs = np.pad(coefs, ((0, 0), (L, L) ), constant_values=((np.nan, np.nan), (np.nan, np.nan)))
    lcoefs = np.pad(lcoefs, ((0, 0), (L, L)), constant_values=((np.nan, np.nan), (np.nan, np.nan)))
   
    # size parameters    
    N = coefs.shape[1]
    Nscale = coefs.shape[0]
     
    # choose the windows (the weights) for average  
    d = np.sqrt(np.linspace(-L, L, 2*L+1)**2)
    if np.char.equal(window,'flat'):
        W=np.ones(2*L+1,)
    elif np.char.equal(window,'bisquare'):
        W = bisquarekernel(np.linspace(-L, L, 2*L+1)/L)
    else:
        W=np.ones(2*L+1)
        print('Not yet implemented --> take flat window instead\n')
    # for grid size
    Wgrid = np.zeros((2*L+1,))
    Wgrid[d<gridsize]=1
    
    # fft of the gridsize window : remove points where it's 0
    Coefs1 = np.copy(coefs[0,:])
    CountValid = NormWithNaNConv1d(Coefs1,Wgrid)
    Index0 = CountValid < 0.5
    # plt.close('all')
    # plt.plot(Coefs1)
    # plt.plot(Wgrid,'r')
    # plt.plot(Index0,'k')
    # print(Coefs1.shape, CountValid.shape)
    # grrr
    # results allocations
    Mean_Std_loc = np.zeros((2,Nscale, N))*np.nan
    Mom12_loc = np.zeros((2,Nscale, N))*np.nan
    Cum12_loc = np.zeros((2,Nscale, N))*np.nan
    Mom4_loc = np.zeros((Nscale, N))*np.nan
    Flatness = np.zeros((Nscale, N))*np.nan
    
    #Mean_loc = np.zeros((Nscale, N))*np.nan
    #Std_loc = np.zeros((Nscale, N))*np.nan
    #Mom1_loc = np.zeros((Nscale, N))*np.nan
    #Mom2_loc = np.zeros((Nscale, N))*np.nan    
    #Cumulant1 = np.zeros((Nscale, N))*np.nan
    #Cumulant2 = np.zeros((Nscale, N))*np.nan
     
    # loop on scales
    for i in range(Nscale):
        print(i,end=' ',flush=True)
        Coefs1 = np.copy(coefs[i,:])
        #LCoefs1 = np.log(abs(Coefs1))
        # normalisation coeficients
        NewNorm1=NormWithNaNConv1d(Coefs1,W)
        Coefs1[np.isnan(Coefs1)]=0
        # compute moments and flatness
        Mean_Std_loc[0, i,L:N-L] = scsig.fftconvolve(Coefs1, W, mode='valid')/NewNorm1
        Mom12_loc[0, i,L:N-L] = scsig.fftconvolve(np.abs(Coefs1), W, mode='valid')/NewNorm1
        Mom12_loc[1, i, L:N-L] = scsig.fftconvolve(Coefs1**2, W, mode='valid')/NewNorm1
        Mom4_loc[i, L:N-L] = scsig.fftconvolve(Coefs1**4, W, mode='valid')/NewNorm1
        Flatness[i, L:N-L] = Mom4_loc[i, L:N-L]/(3*Mom12_loc[1, i, L:N-L]**2)
        Mean_Std_loc[1, i,L:N-L] = Mom12_loc[1,i, L:N-L] - Mean_Std_loc[0, i,L:N-L]**2
        #--
        # Mean_loc[i,L:N-L] = scsig.fftconvolve(Coefs1, W, mode='valid')/NewNorm1
        # Mom1_loc[i,L:N-L] = scsig.fftconvolve(np.abs(Coefs1), W, mode='valid')/NewNorm1
        # Mom2_loc[i, L:N-L] = scsig.fftconvolve(Coefs1**2, W, mode='valid')/NewNorm1
        # Mom4_loc[i, L:N-L] = scsig.fftconvolve(Coefs1**4, W, mode='valid')/NewNorm1
        # Flatness[i, L:N-L] = Mom4_loc[i, L:N-L]/(3*Mom2_loc[i, L:N-L]**2)
        # Std_loc[i, L:N-L] = Mom2_loc[i, L:N-L] - Mean_loc[i,L:N-L]**2
        
        # compute the log of the coeficient (averaged on r)
        LCoefs1 = np.copy(lcoefs[i,:])
        # normalisation
        NewNormL1 = NormWithNaNConv1d(LCoefs1,W)
        LCoefs1[np.isnan(LCoefs1)] = 0
        # compute cumulant
        Cum12_loc[0,i, L:N-L] = scsig.fftconvolve(LCoefs1, W, mode='valid')/NewNormL1        
        Cum12_loc[1,i, L:N -L] = scsig.fftconvolve(LCoefs1**2, W, mode='valid')/NewNormL1
        Cum12_loc[1,i, L:N-L] = Cum12_loc[1,i, L:N-L] - Cum12_loc[0,i, L:N-L]**2
       
        # Cumulant1[i, L:N-L] = scsig.fftconvolve(LCoefs1, W, mode='valid')/NewNormL1        
        # Cumulant2[i, L:N -L] = scsig.fftconvolve(LCoefs1**2, W, mode='valid')/NewNormL1
        # Cumulant2[i, L:N-L] = Cumulant2[i, L:N-L] - Cumulant1[i, L:N-L]**2
        
    print('.')
    
    # remove borders
    Mean_Std_loc = Mean_Std_loc[:, :, L:-L]
    Mom12_loc = Mom12_loc[:, :, L:-L]
    Cum12_loc = Cum12_loc[:, :, L:-L]
    Flatness = Flatness[ :, L:-L]
    # swap axis
    Mean_Std_loc = Mean_Std_loc.transpose((0,2,1))
    Mom12_loc = Mom12_loc.transpose((0,2,1))
    Cum12_loc = Cum12_loc.transpose((0,2,1))
    Flatness = Flatness.T
    # set Nan at empty grid
    Mean_Std_loc[:,Index0,:]=np.nan
    Mom12_loc[:,Index0,:]=np.nan
    Cum12_loc[:,Index0,:]=np.nan
    Flatness[Index0,:]=np.nan
    #print(Mom12_loc.shape, Mean_Std_loc.shape)
    
    # set non valid value (border effect) to nan 
    # Mom1_loc = Mom1_loc[:,L:-L]
    # Mom2_loc = Mom2_loc[:,L:-L]
    # Mom4_loc = Mom4_loc[:,L:-L]
    # Cumulant1 = Cumulant1 [:,L:-L]
    # Cumulant2 = Cumulant2 [:,L:-L]
    
    # swap axes
    # Mom1_loc = Mom1_loc.T
    # Mom2_loc = Mom2_loc.T
    # #Flatness = Flatness.T
    # Cumulant1 = Cumulant1.T
    # Cumulant2 = Cumulant2.T
    #print(np.nanmax(np.abs(Cum12_loc[1,:,:]-Cumulant2)))
    #gf
    # onv veut
    return Mean_Std_loc, Mom12_loc, Flatness ,Cum12_loc
    #return Mom1_loc, Mom2_loc, Flatness ,Cumulant1, Cumulant2

#
def localCorrCoefConv1d(Coefs1sig,Coefs2sig,L, window='flat', gridsize=1):
    '''
        Compute localized correlation coeficients between two sets of values.
        
        rho = localCorrCoefConv1d(Coefs1,Coefs2,L, window)
    
    
        Inputs :
            
            coefs1  : first set of ceoficients 
            
            coefs2  : second set of ceoficients 
           
            L : size of the average (integer stricty positif)
           
            window : weighting function to use. Default : window='flat'.
                     can be window='bisquare'
           
       Outputs :   

            rho : localized correlation coeficients

            
      ##
       P.Thiraux  S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''
    
    assert(Coefs1sig.shape == Coefs2sig.shape)
    # swap axes
    Coefs1sig = Coefs1sig.T
    Coefs2sig = Coefs2sig.T
    Coefs1sig = np.pad(Coefs1sig, ((0, 0), (L, L) ), constant_values=((np.nan, np.nan), (np.nan, np.nan)))
    Coefs2sig = np.pad(Coefs2sig, ((0, 0), (L, L)), constant_values=((np.nan, np.nan), (np.nan, np.nan)))
   
    # size parameters       
    N  = Coefs1sig.shape[1]
    Nscales=Coefs1sig.shape[0]
    
    # choose the windows (the weights) for average  
    d = np.sqrt(np.linspace(-L, L, 2*L+1)**2)
    if np.char.equal(window,'flat'):
        W=np.ones(2*L+1,)
    elif np.char.equal(window,'bisquare'):
        W = bisquarekernel(np.linspace(-L, L, 2*L+1)/L)
    else:
        W=np.ones(2*L+1)
        print('Not yet implemented --> take flat window instead\n')
    
    # for grid size
    Wgrid = np.zeros((2*L+1,))
    Wgrid[d<gridsize]=1
     
    # fft of the gridsize window : remove points where it's 0
    Coefs1 = np.copy(Coefs1sig[0,:])
    CountValid = NormWithNaNConv1d(Coefs1,Wgrid)
    Index0 = CountValid < 0.5
   
    # results allocations    
    rho = np.zeros((Nscales,N))*np.nan
 
    # loop on scales
    for i in range(Nscales):
        # temporary allocation
        Coefs1 = np.copy(Coefs1sig[i,:])
        Coefs2 = np.copy(Coefs2sig[i,:])
        # normalisation
        NewNormprod = NormWithNaNConv1d(Coefs1*Coefs2, W)
        NewNorm1 = NormWithNaNConv1d(Coefs1, W)
        NewNorm2 = NormWithNaNConv1d(Coefs2, W)
        # index of nan
        isnanc1 = np.isnan(Coefs1)
        isnanc2 = np.isnan(Coefs2)
        
        """Calcul de rho local"""        
        Coefs1[np.isnan(Coefs1)]=0
        Coefs2[np.isnan(Coefs2)]=0
        moy1 = scsig.fftconvolve(Coefs1,W,mode = 'valid')/NewNorm1
        moy2 = scsig.fftconvolve(Coefs2,W,mode = 'valid')/NewNorm2
        
        M2_1= scsig.fftconvolve(Coefs1**2,W,mode = 'valid')/NewNorm1
        M2_2 = scsig.fftconvolve(Coefs2**2,W,mode = 'valid')/NewNorm2
        moy_cross = scsig.fftconvolve(Coefs1*Coefs2,W,mode = 'valid')/NewNormprod
        var1 = M2_1 - moy1**2    
        var2 = M2_2 - moy2**2  
        
        rho[i,L:N-L] = (moy_cross - moy1*moy2)/np.sqrt(var1*var2)
        rho[i,N-L:N]=np.nan
        rho[i,0:L]=np.nan
        rho[i,isnanc1]=np.nan
        rho[i,isnanc2]=np.nan
        
        # Non centered
        # rhononcentered[i,L:N-L] = (moy_cross - moy1*moy2)/np.sqrt(M2_1*M2_2)
        # rhononcentered[i,N-L:N]=np.nan
        # rhononcentered[i,0:L]=np.nan
        # rhononcentered[i,isnanc1]=np.nan
        # rhononcentered[i,isnanc2]=np.nan
    # remove border
    rho = rho[ :, L:-L]
    # swap axis
    rho = rho.T
    # set Nan at empty grid
    rho[Index0,:]=np.nan
    
    return rho#, rhononcentered

#%% --------------- Users functions for computing Multiscale coefficient : conv and knn method
# MultiScaleQuantityConv
def MultiScaleQuantityConv(image, radiustot, fftplan=[], theta=[], dtheta=[]):
    """
    Compute multiscale quantities from Image. The image can contain non finite values.
        - box counting of the finite values  N(r) for each pixel
        - Hat wavelet coefs (2* N(r) -N(np.sqrt(2)*r) of the set of finite values 
        - local mean, std, Hat wavelet coeficient and Poor wavelet coefficient 
        of the image (discarding non finite values) 
    
    Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = MultiScaleQuantityConv(image, radiustot, fftplan=[])


    Parameters
    ----------
    data : numpy array of float
        Two dimensional data set of shape  (Nx, Ny).
    radius : numpy array of  float
        vector of radius values of shape (Nr,).
    fftplan : fftw object, optional
        For fast computation we can prepare fftw plan using function
        do_fftplan2d. The default is [].

    Raises
    ------
    TypeError This is for image ONLY.
        The input data is not an image : a numpy array of size (Nx,Ny).

    Returns
    -------
    Count : numpy array of float
        box counting coeficient of the support: numpy array of  shape  (Nr, Nx, Ny).
        It's the number of non nan values in a ball of radius r : N_x(r)
    HatSupport : numpy array of float
        Hat wavelet coeficient of the support: numpy array of  shape  (Nr, Nx, Ny).
        Defined as  2*N_x(r)-N_x(np.sqrt(2)*r)
    Mean : numpy array of float
        Local average (on a ball of size radius) of the mark : numpy array of  shape  (Nr, Nx, Ny).
        M_x(r)
    LMean : numpy array of float
        Local average of the logarithm of the mark : numpy array of  shape  (Nr, Nx, Ny).
    HatMean : numpy array of float
        Hat wavelet coeficient of the mark : numpy array of  shape  (Nr, Nx, Ny).
        Defined as  M_x(r)-M_x(np.sqrt(2)*r)
    LHatMean : numpy array of float
        local Logarithm of Hat wavelet coeficient of the mark : numpy array of  shape  (Nr, Nx, Ny).
    Poor : numpy array of float
        Poor wavelet coeficient of the mark : numpy array of  shape  (Nr, Nx, Ny).
        Defined as  data_x-M_x(r)
    LPoor : numpy array of float
        local Logarithm of  Poor wavelet coeficient of the mark : numpy array of  shape  (Nr, Nx, Ny).      
    Std : numpy array of float
        Local standart deviation (on a ball of size radius)  of the mark.
        numpy array of  shape  (Nr, Nx, Ny). 
    HatStd : numpy array of float
        Local standart deviation (on a ball of size radius)  of the Hat wavelet coeficients. 
        numpy array of  shape  (Nr, Nx, Ny). 

    """
    # check input data
    isimage, ismarked, Nmark = checkinputdata(image)
    if not isimage:
        raise TypeError('This is for image ONLY.')
    
    isAni = False
    if len(dtheta)+len(theta) == 2: # anisotropic case
         isAni = True
    # replace infinite values by nan
    image[np.isinf(image)] = np.nan
    # take care of border effect --> pad by nan nan 
    maxradius = int(np.ceil(np.sqrt(2)*np.max(radiustot)))
    im = np.pad(image, ((maxradius, maxradius) ,(maxradius, maxradius)), constant_values=((np.nan, np.nan),(np.nan, np.nan)))
    
    # make a copy of image with 0 instead of NaN   
    IndNan = np.isnan(im)
    imcopy = np.copy(im)
    imcopy[IndNan] = 0
    # make a mask : one if valid value
    im01 = np.ones(im.shape)
    im01[IndNan] = 0
    
    #soze parameter
    N1,N2 = imcopy.shape
    #Nk=int(np.ceil(np.max(2*radiustot+1)))
    Nborder = int(np.ceil(np.sqrt(2)*np.max(radiustot)))
    Nk = 2*Nborder+1
    
    # define distance for filter
    y, x = np.ogrid[-Nborder:Nborder+1, -Nborder:Nborder+1]
    d = x**2 + y**2
    thetaTot = np.flip(np.arctan2(y,x),axis=0)
    
    # fftw plan
    Nfft1 = N1+Nk-1
    Nfft2 = N2+Nk-1
    
    if len(fftplan)==0:
        a,b, fft_object, br, ffti_br, ffti_object = prepare_plan2d(Nfft1, Nfft2, N_threads)
    else:
        a,b, fft_object, br, ffti_br, ffti_object = fftplan
   

    # allocations
    HatSupport = np.zeros((radiustot.shape[0], N1, N2))
    HatStd = np.zeros((radiustot.shape[0], N1, N2))
    Mean = np.zeros((radiustot.shape[0], N1, N2))
    LMean = np.zeros((radiustot.shape[0], N1, N2))
    HatMean = np.zeros((radiustot.shape[0], N1, N2))
    LHatMean = np.zeros((radiustot.shape[0], N1, N2))
    Poor  = np.zeros((radiustot.shape[0], N1, N2))
    LPoor = np.zeros((radiustot.shape[0], N1, N2)) 
    Std = np.zeros((radiustot.shape[0], N1, N2))#*np.nan 
    Count = np.zeros((radiustot.shape[0], N1, N2))
    
    # loop on scales
    for ir in range(radiustot.shape[0]):
        # %
        print(ir, end=' ', flush = True)
        
        # define first filter
        kernel = np.zeros((Nk,Nk))
        kernel[np.sqrt(d) <= radiustot[ir]]=1
        if isAni:
            kernel=threshkernelangle(kernel,thetaTot,theta,dtheta)
        # kernel[Nborder,Nborder]=0 -> pas de diference
        # fft of the filter
        TFW = np.copy(fft_object(np.pad(kernel, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
        
        # normalisation
        # new --> nickel (no border effect compare to old)
        imtmp = np.pad(im01, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid1 = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]       
        CountValid1[CountValid1 < 0.1] = np.nan # must be integer >0
        Cc = np.copy(CountValid1)
        Cc[IndNan] = np.nan
        
        # sum    
        imtmp = np.pad(imcopy, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        # box counting coeficients
        lamean = lasum / CountValid1
        lamean[IndNan] = np.nan
        # variance of the box counting and poor wavelet (it is the same)
        lasum = convFFTW(imtmp**2, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        M2 = lasum / CountValid1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            lavar = np.maximum(M2-lamean**2,0)
            lastd = np.sqrt(lavar)   
        lastd[IndNan] = np.nan
        
        # define second filter
        kernel = np.zeros((Nk,Nk))
        kernel[(np.sqrt(d) <= np.sqrt(2)*radiustot[ir]) & (np.sqrt(d) > radiustot[ir])] = 1
        # kernel[Nborder,Nborder]=0 -> pas de diference
        # fft of the filter
        TFW = np.copy(fft_object(np.pad(kernel, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
        
        # normalisation
        # new --> nickel (no border effect compare to old)
        imtmp = np.pad(im01, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid2 = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]       
        CountValid2[CountValid2 < 0.1] = np.nan # must be integer >0
        Cc2 = np.copy(CountValid2)
        Cc2[IndNan] = np.nan
        # sum    
        imtmp = np.pad(imcopy, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum2 = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        # box counting coeficients
        lamean2 = lasum2 / CountValid2
        lamean2[IndNan] = np.nan
        
        # variance of the box counting and poor wavelet (it is the same)
        lasum = convFFTW(imtmp**2, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        M2 = lasum / CountValid2
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            lavar = np.maximum(M2 - lamean**2, 0)
            lastd2 = np.sqrt(lavar)              
        lastd2[IndNan] = np.nan
        
        Count[ir,:,:] = Cc
        HatSupport[ir,:,:] = 2 * Cc - Cc2
        Mean[ir,:,:] = lamean
        Poor[ir,:,:] = lamean - im
        HatMean[ir,:,:] = lamean - lamean2
        Std[ir,:,:] = lastd
        HatStd[ir,:,:] = lastd - lastd2
        #% ------ log  of Hat wavelet coeficients
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            LCoefs = np.log(np.abs(HatMean[ir,:,:]))
            
        LCoefs[np.isclose(HatMean[ir,:,:],0)] = np.nan        
        isnan = np.isnan(LCoefs)
        # kernel
        kernel[Nborder, Nborder] = 1 
        TFW = np.copy(fft_object(np.pad(kernel, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
        # normalisation
        tmp = np.ones(LCoefs.shape)
        tmp[isnan] = 0
        imtmp = np.pad(tmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]       
        CountValid[CountValid < 0.1] = np.nan
        
        # sum
        LCoefs[isnan] = 0
        imtmp = np.pad(LCoefs, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        #poor wavelet coeficients 
        cum1=lasum / CountValid
        cum1[isnan] = np.nan
        LHatMean[ir,:,:] = cum1
        
        # ------ log  of the Box Counting coeficients
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            LCoefs = np.log(np.abs(Mean[ir,:,:]))
            
        LCoefs[np.isclose(Mean[ir,:,:],0)] = np.nan        
        isnan = np.isnan(LCoefs)
        # kernel
        kernel[Nborder, Nborder] = 1 
        TFW = np.copy(fft_object(np.pad(kernel, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
        # normalisation
        tmp = np.ones(LCoefs.shape)
        tmp[isnan] = 0
        imtmp = np.pad(tmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]       
        CountValid[CountValid < 0.1] = np.nan
        # sum
        LCoefs[isnan] = 0
        imtmp = np.pad(LCoefs, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        #poor wavelet coeficients 
        cum1 = lasum / CountValid
        cum1[isnan] = np.nan
        LMean[ir,:,:] = cum1       
        
        #% ------ log  of the poor wavelet coeficients
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category = RuntimeWarning)
            LCoefs = np.log(np.abs(Poor[ir,:,:]))
            
        LCoefs[np.isclose(HatMean[ir,:,:],0)] = np.nan        
        isnan = np.isnan(LCoefs)
        # kernel
        kernel[Nborder,Nborder] = 1 
        TFW = np.copy(fft_object(np.pad(kernel, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
        # normalisation
        tmp = np.ones(LCoefs.shape)
        tmp[isnan] = 0
        imtmp = np.pad(tmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]       
        CountValid[CountValid < 0.1] = np.nan
        
        # sum
        LCoefs[isnan] = 0
        imtmp = np.pad(LCoefs, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum = convFFTW(imtmp, TFW, fft_object,ffti_object)[Nborder:Nfft1-Nborder, Nborder:Nfft2-Nborder]
        #poor wavelet coeficients 
        cum1 = lasum / CountValid
        cum1[isnan] = np.nan
        LPoor[ir,:,:] = cum1
        
        
    print('.')
          
    HatMean = HatMean[:, maxradius:-maxradius, maxradius:-maxradius]
    LHatMean = LHatMean[:, maxradius:-maxradius, maxradius:-maxradius]
    Poor = Poor[:, maxradius:-maxradius, maxradius:-maxradius]
    LPoor = LPoor[:, maxradius:-maxradius, maxradius:-maxradius]
    
    Mean = Mean[:, maxradius:-maxradius, maxradius:-maxradius]
    LMean = LMean[:, maxradius:-maxradius, maxradius:-maxradius]
    Std = Std[:, maxradius:-maxradius, maxradius:-maxradius]    
    Count = Count[:, maxradius:-maxradius, maxradius:-maxradius]    
    HatStd = HatStd[:, maxradius:-maxradius, maxradius:-maxradius]    
    HatSupport = HatSupport[:, maxradius:-maxradius, maxradius:-maxradius]    
    
    # Swap the axis : first goes to the last
    
    #return Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd
    #Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = computeHatCoefsConv2d(data, radius,fftplan=fftplan)
    Count = Count.transpose(1, 2, 0)
    HatSupport = HatSupport.transpose(1, 2, 0)
    Mean = Mean.transpose(1, 2, 0)
    LMean = LMean.transpose(1, 2, 0)
    HatMean = HatMean.transpose(1, 2, 0)
    LHatMean = LHatMean.transpose(1, 2, 0)
    Poor = Poor.transpose(1, 2, 0)
    LPoor = LPoor.transpose(1, 2, 0)
    Std = Std.transpose(1, 2, 0)
    HatStd = HatStd.transpose(1, 2, 0)
    
    
    return Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd 

#%
def localMomCumConv2d(coefs,lcoefs,L, window='circle',fftplan=[], gridsize=1):
    '''
        Compute localized statistics (moments and cumulants).
        
        Mom1, Mom2, Flatness ,Cum1, Cum2 = localMomCumConv2d(coefs,lcoefs,L,window)
        
        Inputs :
            
            coefs  : set of ceoficients used to compute moments and flatness
            
            lcoefs:  set of log coefficients used to compute the cumulants
            
            L : size of the average (integer stricty positif)
            
            window : weighting function to use. Default : window='flat'.
                     can be window='bisquare'
            
        Outputs :   
            
            Mom1, Mom2 : localized moment of order one and two
            
            Flatness : localized flatness
            
            cum1, cum2 : localized log cumulant of order one and two
            
        
     ##
      P.Thiraux  S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''
    
    assert(coefs.shape == lcoefs.shape)
    # transpose axis
    coefs = coefs.transpose(2, 0, 1)
    lcoefs = lcoefs.transpose(2, 0, 1)
    #print(coefs.shape, lcoefs.shape)
    
    # take care of border effect --> pad by nan nan 
    imCoefs = np.pad(coefs, ((0, 0), (L, L) ,(L, L)), constant_values=((np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)))
    imLCoefs = np.pad(lcoefs, ((0, 0), (L, L) ,(L, L)), constant_values=((np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)))
    
    # size parameters    
    Nscale, N1, N2 = imCoefs.shape
     
    # choose the windows (the weights) for average  
    y, x = np.ogrid[-L:L+1, -L:L+1]     
    d = np.sqrt( x**2 + y**2 ) /L
    if np.char.equal(window,'square'):
        W = np.ones((2*L+1,2*L+1))
    elif np.char.equal(window,'bisquare'):  
        W = bisquarekernel(d)
    elif np.char.equal(window,'circle'):
        W=np.copy(d)
        W[d<=1]=1
        W[d>1]=0    
    else:
        W = np.ones((2*L+1,2*L+1))
        print('Not yet implemented --> take flat window instead\n')
    # for grid size
    Wgrid = np.zeros((2*L+1,2*L+1))
    Wgrid[d<gridsize/L]=1
    
    
    #print(Wgrid.shape, gridsize)
    #grrr
    # fftw plan
    Nk = 2*L+1
    Nfft1 = N1+Nk-1
    Nfft2 = N2+Nk-1
    #a,b, fft_object, br, ffti_br, ffti_object = prepare_plan2d(Nfft1, Nfft2, N_threads)
    if len(fftplan)==0:
        a,b, fft_object, br, ffti_br, ffti_object = prepare_plan2d(Nfft1, Nfft2, N_threads)
    else:
        a,b, fft_object, br, ffti_br, ffti_object = fftplan
   
     
    # fft of the gridsize window : remove points where it's 0
    TFW=np.copy(fft_object(np.pad(Wgrid, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
    Coefs1 = np.copy(imCoefs[0,:,:])
    iNan=np.isnan(Coefs1)
    imtmp=np.ones((N1,N2))
    imtmp[iNan]=0
    imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
    CountValid=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
    Index0 = CountValid < 0.5
    
    # fft of the window
    TFW=np.copy(fft_object(np.pad(W, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
   
    
    # results allocations
    Mean_Std_loc = np.zeros((2, Nscale, N1, N2))*np.nan
    Mom12_loc = np.zeros((2, Nscale, N1, N2))*np.nan
    Cum12_loc = np.zeros((2, Nscale, N1, N2))*np.nan
    
   # Mean_loc = np.zeros((Nscale, N1, N2))*np.nan
   # Std_loc = np.zeros((Nscale, N1, N2))*np.nan
   # Mom1_loc = np.zeros((Nscale, N1, N2))*np.nan
   # Mom2_loc = np.zeros((Nscale, N1, N2))*np.nan
    Mom4_loc = np.zeros((Nscale, N1, N2))*np.nan
    Flatness = np.zeros((Nscale, N1, N2))*np.nan
    #Cumulant1 = np.zeros((Nscale, N1, N2))*np.nan
    #Cumulant2 = np.zeros((Nscale, N1, N2))*np.nan
     
    # loop on scales
    for i in range(Nscale):
        print(i,end=' ',flush=True)
        Coefs1 = np.copy(imCoefs[i,:,:])
        iNan=np.isnan(Coefs1)
        #LCoefs1 = np.log(abs(Coefs1))
        #
        # normalisation
        # new --> nickel (no border effect compare to old)
        imtmp=np.ones((N1,N2))
        imtmp[iNan]=0
        imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
        CountValid[CountValid < 0.1] = np.nan # must be integer >0
                    
        Coefs1[iNan]=0
        # compute moments and flatness
        imtmp=np.pad(Coefs1, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0]=np.nan
        Mean_Std_loc[0,i,:,:]=lasum/CountValid
        lasum=convFFTW(np.abs(imtmp), TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0]=np.nan
        #Mom1_loc[i,:,:]=lasum/CountValid
        Mom12_loc[0,i,:,:]=lasum/CountValid
        lasum=convFFTW(imtmp**2, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0]=np.nan
        #Mom2_loc[i,:,:]=lasum/CountValid
        Mom12_loc[1,i,:,:]=lasum/CountValid
        Mean_Std_loc [1,i,:,:]= Mom12_loc[1,i,:,:] - Mean_Std_loc [0,i,:,:]**2
        lasum=convFFTW(imtmp**4, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        lasum[Index0]=np.nan
        Mom4_loc[i,:,:]=lasum/CountValid
        
        Flatness[i,:,:]= Mom4_loc[i,:,:]/(3*Mom12_loc[1,i,:,:]**2)
        
        # compute the log of the coeficient (averaged on r)
        # normalisation
        LCoefs1 = np.copy(imLCoefs[i,:,:])
        iNan=np.isnan(LCoefs1)
        imtmp=np.ones((N1,N2))
        imtmp[iNan]=0
        imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
        CountValid[CountValid < 0.1] = np.nan # must be integer >0
        # 
        LCoefs1[np.isnan(LCoefs1)] = 0
        imtmp=np.pad(LCoefs1, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        lasum[Index0]=np.nan
        Cum12_loc[0,i,:,:]=lasum/CountValid
        lasum=convFFTW(imtmp**2, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        lasum[Index0]=np.nan
        Cum12_loc[1,i,:,:]=lasum/CountValid-Cum12_loc[0,i,:,:]**2
        
    print('.')
    
    Mean_Std_loc = Mean_Std_loc[:,:,L:-L,L:-L]
    Mom12_loc = Mom12_loc[:,:,L:-L,L:-L]
    Cum12_loc = Cum12_loc[:,:,L:-L,L:-L]
    #Mean_loc = Mean_loc[:,L:-L,L:-L]
    #Std_loc = Std_loc[:,L:-L,L:-L]
    #Mom1_loc = Mom1_loc[:,L:-L,L:-L]
    #Mom2_loc = Mom2_loc[:,L:-L,L:-L]
    ##Mom4_loc = Mom4_loc[:,L:-L,L:-L]
    Flatness = Flatness[:,L:-L,L:-L]
    #Cumulant1 = Cumulant1[:,L:-L,L:-L]
    #Cumulant2 = Cumulant2[:,L:-L,L:-L]
    
    # swap axes
    Mean_Std_loc = Mean_Std_loc.transpose(0,2,3,1)
    Mom12_loc = Mom12_loc.transpose(0,2,3,1)
    Cum12_loc = Cum12_loc.transpose(0,2,3,1)
    Flatness = Flatness.transpose(1,2,0)
    return Mean_Std_loc, Mom12_loc, Flatness ,Cum12_loc
#
def localCorrCoefConv2d(Coefs1sig,Coefs2sig,L, window='flat',fftplan=[], gridsize=1):
    '''
        Compute localized correlation coeficients between two sets of values.
        
        rho = localCorrCoefConv2d(Coefs1,Coefs2,L, window)
    
    
        Inputs :
            
            coefs1  : first set of ceoficients 
            
            coefs2  : second set of ceoficients 
           
            L : size of the average (integer stricty positif)
           
            window : weighting function to use. Default : window='flat'.
                     can be window='bisquare'
           
       Outputs :   

            rho : localized correlation coeficients

            
      ##
       P.Thiraux  S. Roux, ENS Lyon, June 2022,  stephane.roux@ens-lyon.fr

    '''
    
    assert(Coefs1sig.shape == Coefs2sig.shape)
    # transpose axis
    Coefs1sig = Coefs1sig.transpose(2, 0, 1)
    Coefs2sig = Coefs2sig.transpose(2, 0, 1)
    
    # take care of border effect --> pad by nan nan 
    imCoefs1 = np.pad(Coefs1sig, ((0, 0), (L, L) ,(L, L)), constant_values=((np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)))
    imCoefs2 = np.pad(Coefs2sig, ((0, 0), (L, L) ,(L, L)), constant_values=((np.nan, np.nan), (np.nan, np.nan), (np.nan, np.nan)))
    
    # size parameters       
    Nscale, N1, N2 = imCoefs1.shape
     
    # choose the windows (the weights) for average  
    if np.char.equal(window,'flat'):
        W = np.ones((2*L+1,2*L+1))
        y, x = np.ogrid[-L:L+1, -L:L+1]     
        d = np.sqrt( x**2 + y**2 ) /L
    elif np.char.equal(window,'bisquare'):
        y, x = np.ogrid[-L:L+1, -L:L+1]     
        d = np.sqrt( x**2 + y**2 ) /L
        W = bisquarekernel(d)
    else:
        y, x = np.ogrid[-L:L+1, -L:L+1]     
        d = np.sqrt( x**2 + y**2 ) /L
        W = np.ones((2*L+1,2*L+1))
        print('Not yet implemented --> take flat window instead\n')
     
    # for grid size
    Wgrid = np.zeros((2*L+1,2*L+1))
    Wgrid[d<gridsize/L]=1
    
    # fftw plan
    Nk = 2*L+1
    Nfft1 = N1+Nk-1
    Nfft2 = N2+Nk-1
    if len(fftplan)==0:
        a,b, fft_object, br, ffti_br, ffti_object = prepare_plan2d(Nfft1, Nfft2, N_threads)
    else:
        a,b, fft_object, br, ffti_br, ffti_object = fftplan
    #a,b, fft_object, br, ffti_br, ffti_object = prepare_plan2d(Nfft1, Nfft2, N_threads)

    # fft of the gridsize window : remove points where it's 0
    TFW=np.copy(fft_object(np.pad(Wgrid, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
    Coefs1 = np.copy(imCoefs1[0,:,:])
    iNan=np.isnan(Coefs1)
    imtmp=np.ones((N1,N2))
    imtmp[iNan]=0
    imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
    CountValid=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
    #Index0=CountValid < 0.5
    Coefs2 = np.copy(imCoefs2[0,:,:])
    iNan=np.isnan(Coefs1)
    imtmp=np.ones((N1,N2))
    imtmp[iNan]=0
    imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
    CountValid2=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
    Index0=(CountValid < 0.5) & (CountValid2 < 0.5)
    
   
    # fft of the window
    TFW=np.copy(fft_object(np.pad(W, ((0, Nfft1-Nk) ,(0, Nfft2-Nk)), constant_values=((0, 0),(0,0)))))
    
    # results allocations    
    rho = np.zeros((Nscale, N1, N2))*np.nan
    #rhononcentered = np.zeros((len(echelle),N))*np.nan
    
    # loop on scales
    for i in range(Nscale):
        print(i,end=' ',flush=True)
        # temporary allocation
        Coefs1 = np.copy(imCoefs1[i,:,:])
        Coefs2 = np.copy(imCoefs2[i,:,:])
        
        iNan1 = np.isnan(Coefs1)
        iNan2 = np.isnan(Coefs2)
        
        
        # Normalisation
        imtmp=np.ones((N1,N2))
        imtmp[iNan1]=0
        imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid1=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
        CountValid1[CountValid1 < 1e-5] = np.nan # must be integer >0
        imtmp=np.ones((N1,N2))
        imtmp[iNan2]=0
        imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid2=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
        CountValid2[CountValid2 < 1e-5] = np.nan # must be integer >0
        imtmp=np.ones((N1,N2))
        imtmp[iNan1]=0
        imtmp[iNan2]=0
        imtmp=np.pad(imtmp, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))
        CountValid12=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]       
        CountValid12[CountValid12 < 1e-5] = np.nan # must be integer >0
        
        Coefs1[iNan1]=0
        Coefs2[iNan2]=0
        # compute moments 
        imtmp=np.pad(Coefs1, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0] = np.nan
        moy1=lasum/CountValid1
        
        #imtmp=np.pad(Coefs1, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))   
        lasum=convFFTW(imtmp**2, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0] = np.nan
        M2_1=lasum/CountValid1
        var1 = M2_1 - moy1**2    
        var1[var1< 1e-5]=np.nan  
        #plt.clf()
        #plt.imshow(Coefs1)
        #grrr
        imtmp=np.pad(Coefs2, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0] = np.nan
        moy2=lasum/CountValid2
        #imtmp=np.pad(Coefs2, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum=convFFTW(imtmp**2, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        M2_2=lasum/CountValid2
        var2 = M2_2 - moy2**2   
        var2[var2< 1e-5]=np.nan  
        #print(np.nanmin(var1),np.nanmin(var2))
        imtmp=np.pad(Coefs1*Coefs2, ((0, Nk-1) ,(0, Nk-1)  ), constant_values=((0, 0),(0,0)))      
        lasum=convFFTW(imtmp, TFW, fft_object,ffti_object)[L:Nfft1-L,L:Nfft2-L]
        #lasum[lasum==0]=np.nan
        lasum[Index0] = np.nan
        moy_cross=lasum/CountValid12
        #moy_cross[iNan1]=np.nan
        #moy_cross[iNan2]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rho[i,:,:] = (moy_cross - moy1*moy2)/np.sqrt(var1*var2)
           
        #rho[i,:,:] = (moy_cross - moy1*moy2)/np.sqrt(var1*var2)
        #print(rho.shape)
        # Non centered
        # rhononcentered[i,L:N-L] = (moy_cross - moy1*moy2)/np.sqrt(M2_1*M2_2)
        # rhononcentered[i,N-L:N]=np.nan
        # rhononcentered[i,0:L]=np.nan
        # rhononcentered[i,isnanc1]=np.nan
        # rhononcentered[i,isnanc2]=np.nan
    print('.')
    rho = rho[:,L:-L,L:-L]
    # swap axes
    rho = rho.transpose(1,2,0)
    return rho#, rhononcentered
#%% knn----------
# %%
def computeGridValue(tree,Count,IndexNear,index2,mygrid, radius, Tloc, weightingfunction):
    """

    NWmean, NWstd, Wmean, Wstd, Mom, Cum = computeGridValue(tree,Count,IndexNear,index,mygrid, radius, T, k=0):

    For marked and non-marked point process
    Compute weighted statistics of the number of point location or value.
    The (geographical) weighting is obtained using the bi-squared function of size T.
       
    Input :

       tree      - a kdtree obtained on the IndexNear points (a subset of the Npts total points)
       Count     - matrix of size Nptsxlength(radius) with wavelet coef
       IndexNear - location of the points used for the kdtree
       index     - index of the grid points under investigations
       mygrid    - all the grid points (size (Ngridpts, 2) with X and Y spatial location
       radius   - list of scales to be investigated
       T        - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       k        - number of min neighbors for the adaptive kernel smoothing. If k=0 : fixed kernel.
        

     Output :

       res - list of two dimensional numpy matrix (of size length(data) X length(radius)

              NWmean  : non weighted mean
              NWstd   : non weighted standart deviation
              Wmean   : weighted mean
              Wstd    : weighted standart deviation
              Mom     : weighted moment (order 0 to 4) of the absolute value of the coefficient. 
              Cum     : weighted cumulant (order one and two) of the coefficient. 


    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
     S.G.  Roux, ENS Lyon, November 2021
    """

    # make the tree accordingly to k
    Tmax = np.nanmax(Tloc)
    
    neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index2,:], r = Tmax, count_only=False, return_distance=True, sort_results=False)

    
    print('.',end='')
    sys.stdout.flush()
 
    # non weighted coef
    tmp_fixed2=[ Count[IndexNear[neighbors_i_fixed[igrid]],:] for igrid in range (len(index2))]
    # number of  coef
    ltmp=np.array([ len(dist_ie_fixed[igrid]) for igrid in range (len(index2))])
    # the weight
    Wfinal2=[  weightingfunction(dist_ie_fixed[igrid], Tloc[igrid]) for igrid in range (len(index2))]
    # compute D0 and moments
    D0 = np.array([(np.nansum( Wfinal2[igrid]*np.abs(tmp_fixed2[igrid])**-1  , axis=0)) for igrid in range(len(index2))])
    D0 = D0**-1
    print('.',end='')
    #Mom1 = np.array([np.nansum( np.abs(wtmp_fixed2[igrid]) , axis=0)  for igrid in range(len(index2))])
    Mom1 = np.array([np.nansum( np.abs(tmp_fixed2[igrid]) * Wfinal2[igrid], axis=0)  for igrid in range(len(index2))])
    print('.',end='')
    #Mom2 = np.array([np.nansum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]), axis=0)  for igrid in range(len(index2))])
    Mom2 = np.array([np.nansum( tmp_fixed2[igrid]**2 * Wfinal2[igrid], axis=0)  for igrid in range(len(index2))])
    #Std = Mom2 - Mean**2
    print('.',end='')
    #Mom3 = np.array([np.nansum( tmp_fixed2[igrid]**3 * Wfinal2[igrid], axis=0)  for igrid in range(len(index2))])
    print('.',end='')
    #Mom4 = np.array([np.nansum( np.abs(wtmp_fixed2[igrid] * tmp_fixed2[igrid]**3), axis=0)  for igrid in range(len(index2))])
    Mom4 = np.array([np.nansum( tmp_fixed2[igrid]**4 * Wfinal2[igrid], axis=0)  for igrid in range(len(index2))])
    print('. ',end='')
    sys.stdout.flush()
    #Mom=np.stack([D0, Mom1, Mom2, Mom3, Mom4],axis=0)
    
    # compute Non weighted std
    newdist=[ localdist(dist_ie_fixed[igrid],Tloc[igrid])  for igrid in range(len(index2))]
   
    NWstd = np.array([np.nanstd( tmp_fixed2[igrid]*newdist[igrid][:,np.newaxis], axis=0, ddof=1) for igrid in range(len(index2))])
    NWstd[ltmp<=10,:] =0
    NWmean = np.array([np.nanmean( tmp_fixed2[igrid]*newdist[igrid][:,np.newaxis], axis=0) for igrid in range(len(index2))])
    NWmean[ltmp<=10,:]=0          
    print('*',end='')
    sys.stdout.flush()
    
    # compute weighted mean and std
    #average = np.array([np.average(tmp_fixed2[igrid], weights=Wfinal2[igrid], axis=0) for igrid in range(len(index2))])
    average = np.array([ np.nansum(tmp_fixed2[igrid]*Wfinal2[igrid], axis=0) for igrid in range(len(index2))])
    Wmean = average
    Wmean[ltmp<=10,:]=0      
    #Wstd  = np.array([np.average((tmp_fixed2[igrid]-average[igrid])**2, weights=Wfinal2[igrid], axis=0) for igrid in range(len(index2))])
    Wstd  = np.array([np.nansum((tmp_fixed2[igrid]-average[igrid])**2*Wfinal2[igrid], axis=0) for igrid in range(len(index2))])
    Wstd[ltmp<=10,:]=0
    Wstd = np.sqrt(Wstd)
    print('* ',end='')
    
    # compute Weighted cumulant
    # need to remove the 0 before taking the log on coefs and weight
    average= np.array([logaverage(tmp_fixed2[igrid], Wfinal2[igrid]) for igrid in range(len(index2))])    
    #average = np.array([np.average(np.log(np.abs(tmp_fixed2[igrid])), weights=Wfinal2[igrid], axis=0) for igrid in range(len(index))])
    Cum1 = average
    Cum1[ltmp<=10,:]=0     
    print('+',end='')   
    Cum2=np.array([logaverage2(tmp_fixed2[igrid],average[igrid], Wfinal2[igrid]) for igrid in range(len(index2))])
    Cum2[ltmp<=10,:]=0   
    print('+ ',end='')
    
    Mean_std=np.stack([Wmean, Wstd], axis = 0)
    Mom12=np.stack([ Mom1, Mom2],axis=0)
    Cum=np.stack([Cum1, Cum2],axis = 0)
    Flatness=Mom4/(3*Mom2**2)
    sys.stdout.flush()   
    
    #print(NWmean.shape, NWstd.shape, Wmean.shape, Wstd.shape, Mom.shape, Cum.shape)
    #grrrr
    # la sortie
    # Mean_Std_loc, Mom12_loc, Flatness ,Cum12_loc
    #return NWmean, NWstd, Wmean, Wstd, Mom, Cum
    #print('TTTT ',NWmean.shape, Mean_std.shape)
    return Mean_std, Mom12, Flatness, Cum
# %%

# %% Local Multiscale Analysis
# add Tloc and function
def LmsAnalysis(data,Count,X,Y,radius, T, adaptive = False, weights = 'flat',  Nanalyse=2**16, Tloc=[]):
    """

    res = WaveSmoothingOptim(data,Wave,X,Y,radius,T,Nanalyse=2**16, k = 0, isvalued=0))

    For marked and non-marked point process
    Compute kernel smoothing of the wavelet coefficient of a dataset of points.
    The geographical weighting is obtained using the bi-squared function of size T.
   
    Input :

       data     - matrix of size Nx2 --> position (x,y) for N points
       Wave     - matrix of size Nxlength(radius) with wavelet count
                  Can be obtained using the function  GWFA_count.m
       X        - array of dim 2 with x-postion of the grid nodes
       Y        - array of dim 2 with y-postion of the grid nodes : X and Y must have the same size
       radius   - list of scales to be investigated
       T        - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       Nanalyse - number of points to analyse in one bach. Default is 2**16 points.
       G        - mean of all pilot density estimates
       isvalued - boolean. If = 1 remove the grid point where the value of the point inside thegrid pixel are all 0.
       

     Output :

       res - list of two dimensional numpy matrix (of size length(data) X length(radius)

              res[Ø] = Wmean   : weighted mean
              res[1] = Wstd    : weighted standart deviation
              res[2] = NWratio : non weighted mean
              res[3] = NWstd   : non weighted standart deviation
              res[4] = Mom     : weighted moment (order 0 to 4) of the absolute value
                      of thecoefficient. matrix of size 5 X length(data) X length(radius)
              res[5] = Cum     : weighted cumulant one and two 
                      of thecoefficient. matrix of size 2 X length(data) X length(radius)



    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
     S.G.  Roux, ENS Lyon, November 2021
    """


    si=data.shape
    if si[1] < 2 or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')
        
    if Count.ndim > 2:
        raise TypeError('The second arument must be two or threedimensional.')

    if Count.shape[0] != si[0]:
        raise TypeError('The twofirst arument must have the same length.')

    if radius[radius>T].shape[0] >0:
        raise TypeError('The last argument must be greater than a sqrt(2)*radius.')

    # check grid
    if np.sum(np.abs(np.array(X.shape)-np.array(Y.shape)))>0:
        raise TypeError('X and Y must have the same size.')

    if np.sum(np.diff(X,axis=1,n=2))>0:
        raise TypeError('X must be regulary sampled.')
    else:
        gridsizex=max(X[0][1] - X[0][0],X[1][0] - X[0][0])
        
    if np.sum(np.diff(Y,axis=0,n=2))>0:
        raise TypeError('Y must be regulary sampled.')
    else:
        gridsizey=max(Y[1][0] - Y[0][0],Y[0][1] - Y[0][0])
    
    if gridsizex != gridsizey:
        raise TypeError('X and Y must have same sampling.')
    
    if radius.size != Count.shape[1]:
        raise TypeError('The size of the second arument must be [N,R,M] where R is the number of scales (=radius.size)')
     
    
    # create grid points
    mygrid=np.column_stack([X.flatten(),Y.flatten()])
    gridlength=mygrid.shape[0]
    gridsizecurr = gridsizex
    
    Nanalyse2=300000
    if Nanalyse==0:
        print('Warning : dangerous if your set of data is large.')
        Nanalyse=gridlength
        Nbunch=1
    else: # find how many bunchs       
        Nbunch = np.ceil(gridlength/Nanalyse).astype(int)
    
    #print('Computing in {:d} bunch(es)'.format(Nbunch))
    # choose weighting
    
    if isinstance(weights, str):
        if weights == 'flat':
            weightingfunction = partial(geographicalWeight,func = flatWindow)
            print('No weight')
        elif weights == 'Epanechnikov':
            weightingfunction = partial(geographicalWeight,func = EpanechnikovWindow)
        elif weights == 'bisquare':
            weightingfunction = partial(geographicalWeight,func = bisquareWindow)
        elif weights == 'triangular':
            weightingfunction = partial(geographicalWeight,func = triangularWindow)
        elif weights == 'tricube':
            weightingfunction = partial(geographicalWeight,func = tricubeWindow)
            
    elif isinstance(weights, types.FunctionType):
        weightingfunction = partial(geographicalWeight,func = weights)
    else:
        raise TypeError('ERROR : weighting is of unknown type.).')
        
    
    # results allocations    
    #Mom = np.nan*np.zeros( (5, gridlength,radius.size), dtype=float)
    Cum = np.nan*np.zeros( (2, gridlength,radius.size), dtype=float)
    #NWstd = np.nan*np.zeros( (gridlength,radius.size), dtype=float)
    #Wstd = np.nan*np.zeros( (gridlength,radius.size), dtype=float)
    #NWmean= np.nan*np.zeros( (gridlength,radius.size), dtype=float)
    #Wmean = np.nan*np.zeros( (gridlength,radius.size), dtype=float)
    # new alloc :
    Mean_Std = np.nan*np.zeros( (2, gridlength,radius.size), dtype=float)
    Mom12 = np.nan*np.zeros( (2, gridlength,radius.size), dtype=float)
    Flatness = np.nan*np.zeros( (gridlength,radius.size), dtype=float)
    if adaptive == True:
        if len(Tloc) ==0: # compute local env
            print('ADAPTIVE KERNEL: Computing pilot density')
            CountT= localWaveTrans(data[:,(0,1)], np.atleast_1d(T))
            #out_T = LmsAnalysis(data[:,(0,1)],CountT,X,Y,np.atleast_1d(T), T=T,Nanalyse=2**9)
            #Wmean_T, Wstd_T, NWmean_T, NWstd_T, Mom_T, Cum_T = out_T
            WmeanStd_T, Mom_T, Flat_T, Cum_T = LmsAnalysis(data[:,(0,1)],CountT,X,Y,np.atleast_1d(T), T=T,Nanalyse=2**9)
            Wmean_Tloc = WmeanStd_T[0,:,0]
            G = np.nanmean(WmeanStd_T[0,:,0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                lambda_ie = G/Wmean_Tloc
            Tloc = lambda_ie*T
            Tlocmax = np.nanquantile(Tloc,0.8)
            Tloc[Tloc>Tlocmax] = Tlocmax
            
            T=Tlocmax
            print('ADAPTIVE KERNEL computed.')
        elif X.flatten().shape[0] == Tloc.flatten().shape[0]:
            T=np.nanmax(Tloc)
        else: 
            raise TypeError('ERROR : wrong size for the local environment')
        #Tloc=Tloc*0+200
    else:
        Tloc=T*np.ones(X.size)
        
    # Loop on bunch
    for ibunch in range(Nbunch):
        # %
        print('bunch {:d}/{:d} '.format(ibunch+1,Nbunch), end=' ')
        sys.stdout.flush()
        # get data index of current bunch
        index=np.arange(ibunch*Nanalyse,(ibunch+1)*Nanalyse,1)
        index=index[index<gridlength]

        # we restrict the tree to points whithin a radius T (which must be > radius2)
        mi=np.min(mygrid[index,:], axis=0)
        ma=np.max(mygrid[index,:], axis=0)
        #print('ADAPTIVE KERNEL STEP 2: Computing local analysis')
        IndexNear=np.where((data[:,0] >mi[0]-T) & (data[:,0] <ma[0]+T) & (data[:,1]  >mi[1]-T) & (data[:,1] <ma[1]+T))[0]
         
        # make the tree with the nearest points only
        if IndexNear.shape[0]>0:
                #print('coucou',len(IndexNear))
                tree = KDTree(data[IndexNear,0:2])
                Idxtot = tree.query_radius(mygrid[index,:], r = np.sqrt(2)*gridsizecurr, count_only=False, return_distance=False)
                # find and remove empty (no point location) grid pixels
                thelengths=np.array([ Idxtot[igrid].shape[0] for igrid in range (len(Idxtot))])
                IdxMin, = np.where(thelengths>0.)
                index = index[IdxMin]
                Idxtot=Idxtot[IdxMin]
                thelengths=thelengths[IdxMin]
                
                # find and remove grid pixel with only 0 value
                # if isvalued:            
                #     tmp_fixed=np.array([ np.nansum(np.abs(Count[IndexNear[Idxtot[igrid]],:])) for igrid in range (len(Idxtot))])           
                #     IdxMin, = np.where(tmp_fixed>0.)
                #     index = index[IdxMin]
                #     Idxtot=Idxtot[IdxMin]
                #     thelengths=thelengths[IdxMin]
                    
                # Managed the number of grid points  (too many neighboors for some)
                cumsumbunch=np.cumsum(thelengths)
                #print(cumsumbunch)
                
                Nflowers=int(np.ceil(np.sum(thelengths)/Nanalyse2))
                print('{:d} batch(s).'.format(Nflowers))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    
                    #start = time.process_time()
                    for iflower in range(Nflowers):
                        i1,=np.where(iflower*Nanalyse2 <= cumsumbunch)
                        i2,=np.where(cumsumbunch <  (iflower+1)*Nanalyse2)
                        flowers=np.intersect1d(i1,i2)
                        #print('Mean of adaptive T: ', np.nanmean(Tloc))
                        #print('uu ',Tloc.shape,flowers.shape)
                        #NWmean[index[flowers],:], NWstd[index[flowers],:], Wmean[index[flowers],:], Wstd[index[flowers],:], Mom[:,index[flowers],:], Cum[:,index[flowers],:]=computeGridValueAdapt2(tree,Count,IndexNear,index[flowers],mygrid, radius,Tloc[index[flowers]],weightingfunction)
                        Mean_Std[:,index[flowers],:], Mom12[:,index[flowers],:], Flatness[index[flowers],:],Cum[:,index[flowers],:]=computeGridValue(tree,Count,IndexNear,index[flowers],mygrid, radius,Tloc[index[flowers]],weightingfunction)
                
                print('')
                # pack output in list of arrays   
        #out=[Wmean, Wstd, NWmean, NWstd, Mom, Cum]
        #out=[Mean_Std, Mom12, Flatness, Cum]
        #print(Wmean.shape,Mean_Std.shape)
    return Mean_Std, Mom12, Flatness, Cum

# MultiScaleQuantitySupportKnn
def MultiScaleQuantitySupportKnn(data,radius, Nanalyse=2**14,destination=np.array([]), NonUniformData=False):
    """
     Compute multiscale quantities of the support of  a list of coordinate. 
         - box counting of the finite values  N(r) for each point.
         - Hat wavelet coefs (2* N(r) -N(np.sqrt(2)*r) of the set of finite values 
                                  
    
    Count, HatSupport = MultiScaleQuantitySupportKnn(data,radius, Nanalyse=2**14,destination=np.array([]), NonUniformData=False)
    
    Parameters
    ----------
    data : numpy array of float
        Two dimensional data set of shape  (N, 2) containing the position x and y of N points.
    radius : numpy array of  float
        vector of radius values of shape (Nr,).
    Nanalyse : integer, optional
        Size of the batch. The default is 2**14.
    destination : TYPE, optional
        DESCRIPTION. The default is np.array([]).
    NonUniformData : boolean, optional
        Set to true if the data are not uniformly sampled is space. 
        If true, run faster using optimized batch contents.
        The default is False.

    Raises
    ------
    TypeError 'This is for point process ONLY.
        The data is certainly an image and this function is ONLY for non uniformly sampled data.

    Returns
    -------
    Count : numpy array of float
        Box counting of the number of points in a ball of radius radius around each data points.
        Count.shape = (N, Nr).
    HatSupport : numpy array of float
        Hat wavelet coefficients of the support : 2 * N(r) - N(sqrt(2) * r ).
        Count.shape = (N, Nr).

    """
    # check input data
    isimage, ismarked, Nmark = checkinputdata(data)
    if isimage:
        raise TypeError('This is for point process ONLY.')
    
    # we compute Box Counting B(r) nad B(r)-B(sqrt(2)*r)
    Count1 = localWaveTrans(data, radius, Nanalyse = Nanalyse, destination = destination, NonUniformData = NonUniformData)
    Count2 = localWaveTrans(data, radius*np.sqrt(2), Nanalyse = Nanalyse, destination = destination, NonUniformData = NonUniformData, verbose = False)
    
    Count = Count1
    HatSupport = 2 * Count1 - Count2
        
    return Count, HatSupport

# MultiScaleQuantityMarkKnn
def MultiScaleQuantityMarkKnn( data, radius, Nanalyse = 2**14, destination = np.array([]), NonUniformData=False):
    # check input data
    isimage, ismarked, Nmark = checkinputdata( data)
    if isimage:
        raise TypeError('This is for point process ONLY.')
    
    Poor1, Count1 = localWaveTrans(data, radius, Nanalyse = Nanalyse)
    Poor2, Count2 = localWaveTrans(data, radius * np.sqrt(2), Nanalyse = Nanalyse, verbose = False)
    
    print(' ---- Computing log of coefs.')
    Count = Count1[ :, :, 0]
    HatSupport = 2 * Count1[ :, :, 0] - Count2[ :, :, 0]
    Mean = Count1[ :, :, 1]
    LMean = locallogCoefs(data[ :, ( 0, 1)], Mean, radius, Nanalyse = 2**14, verbose = False)
    HatMean = Count1[ :, :, 1] - Count2[ :, :, 1]
    LHatMean = locallogCoefs(data[:, (0, 1)], HatMean, radius, Nanalyse = 2**14, verbose = False)
    HatStd = Count1[:,:,2] - Count2[:,:,2]
    Poor = Poor1[ :, :, 1]   
    LPoor = locallogCoefs( data[:, (0, 1)], Poor, radius, Nanalyse = 2**14, verbose = False)
    Std = Count1[ :, :, 2]
    
    
    return Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd   
        
#  MultiScaleQuantityPP
def MultiScaleQuantityPP(data,radius, Nanalyse=2**14,destination=np.array([]), NonUniformData=False, fftplan=[]):
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    radius : TYPE
        DESCRIPTION.
    Nanalyse : TYPE, optional
        DESCRIPTION. The default is 2**14.

    Returns
    -------
    Count : TYPE
        DESCRIPTION.
    HatSupport : TYPE
        DESCRIPTION.
    HatMean : TYPE
        DESCRIPTION.
    HatStd : TYPE
        DESCRIPTION.
    Poor : TYPE
        DESCRIPTION.
    LogPoor : TYPE
        DESCRIPTION.

    """
    # check input data
    isimage, ismarked, Nmark = checkinputdata(data)
   
    # choose method
    if (ismarked == 0) & (not isimage) :        
        
        #waveletKnnSupport
        Count, HatSupport = MultiScaleQuantitySupportKnn(data,radius, Nanalyse=Nanalyse,destination=destination, NonUniformData=NonUniformData)
        
        Mean = np.array([])
        LMean = np.array([])
        HatMean = np.array([])
        LHatMean = np.array([])
        Poor = np.array([])
        LPoor = np.array([])
        Std = np.array([])
        HatStd = np.array([])
        
    
    else: # marked points
        if isimage:
            #grrr
            #Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = computeHatCoefsConv2d(data, radius)
            Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = MultiScaleQuantityConv(data, radius, fftplan=[])
        else:
            #hhhhh
            Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd  = MultiScaleQuantityMarkKnn(data,radius, Nanalyse=Nanalyse,destination=destination, NonUniformData=NonUniformData)
            
            
                
    return Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd

def MultiScaleQuantityConv1d(data, radius, wavelet ='poor'):
    if wavelet == 'harr':
        Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd  = computeHarrCoefsConv1d(data, radius)
    else:# Hat
        Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = computeCoefsConv1d(data, radius)
    # manque moyenne, log de la moyenne , std et std Hat
    #elif wavelet == 'harr':
    #elif wavelet == 'hat':
    
    return Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd
#%%
# %%
#def computeGridValuebivarSS(tree,Count,IndexNear,index,mygrid, radius, T,k):
def computeGridValuebivar(tree, Count1, Count2, IndexNear, index, mygrid, radius, Tloc, weightingfunction):
    """

    rhoss = computeGridValuebivarSS(tree,Count,IndexNear,index,mygrid, radius, T, k=0):

    For marked and non-marked point process
    Compute weighted statistics of the number of point location or value.
    The (geographical) weighting is obtained using the bi-squared function of size T.
       
    Input :

       tree      - a kdtree obtained on the IndexNear points (a subset of the Npts total points)
       Count     - matrix of size Nptsxlength(radius) with wavelet coef
       IndexNear - location of the points used for the kdtree
       index     - index of the grid points under investigations
       mygrid    - all the grid points (size (Ngridpts, 2) with X and Y spatial location
       radius   - list of scales to be investigated
       T        - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       k        - number of min neighbors for the adaptive kernel smoothing. If k=0 : fixed kernel.
        

     Output :

       rhoss - self similar correlation coefficient (of size length(data) X length(radius)


    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
     S.G.  Roux, ENS Lyon, November 2021
    """
    
    Tmax = np.nanmax(Tloc)
    neighbors_i_fixed, dist_ie_fixed = tree.query_radius(mygrid[index,:], r = Tmax, count_only=False, return_distance=True, sort_results=False)
    print('.',end='')
    sys.stdout.flush()

    # non weighted coef
    tmp_fixed1=[ Count1[IndexNear[neighbors_i_fixed[igrid]],:] for igrid in range (len(index))]
    tmp_fixed2=[ Count2[IndexNear[neighbors_i_fixed[igrid]],:] for igrid in range (len(index))]

    # number of  coef
    #ltmp=np.array([ len(dist_ie_fixed[igrid]) for igrid in range (len(index))])
    # the weight
    Wfinal=[  weightingfunction(dist_ie_fixed[igrid], Tloc[igrid]) for igrid in range (len(index))]
   
    m01=np.array([ np.nansum(tmp_fixed1[igrid] * Wfinal[igrid], axis=0) for igrid in range(len(index))])
    m10=np.array([ np.nansum(tmp_fixed2[igrid] * Wfinal[igrid], axis=0) for igrid in range(len(index))])   
    m02=np.array([ np.nansum(tmp_fixed1[igrid]**2 * Wfinal[igrid], axis=0) for igrid in range(len(index))])
    m20=np.array([ np.nansum(tmp_fixed2[igrid]**2 * Wfinal[igrid], axis=0) for igrid in range(len(index))])
    m11=np.array([ np.nansum(tmp_fixed1[igrid] * tmp_fixed2[igrid] * Wfinal[igrid], axis=0) for igrid in range(len(index))])   
    var1=m02-m01**2
    var2=m20-m10**2
    rho= (m11-m01*m10)/np.sqrt(var1*var2)
    
    return rho
# 

# %% WaveSmoothingOptim for bivariate statistique
def localCorrCoefKnn(data, Count1, Count2, X, Y, radius, T, adaptive=False, weights = 'flat', Nanalyse=2**16, Tloc=[]):
#(datapos, Coefs1, Coefs2, X, Y,radius, T,  Nanalyse=2**16, k = 0):
    """

   res = WaveSmoothingBivar(data,Wave,X,Y,radius,T,Nanalyse=2**16, k = 0)

   For marked and non-marked point process
   Compute kernel smoothing of the wavelet coefficient of a dataset of points.
   The geographical weighting is obtained using the bi-squared function of size T.
   Only for non valued analysis (non marked process)
   
   Input :

       data     - matrix of size Nx2 --> position (x,y) for N points
       Wave     - matrix of size Nxlength(radius) with wavelet count
                  Can be obtained using the function  GWFA_count.m
       X        - array of dim 2 with x-postion of the grid nodes
       Y        - array of dim 2 with y-postion of the grid nodes : X and Y must have the same size
       radius   - list of scales to be investigated
       T        - bandwith of fixed kernel OR distance upper boundary of adaptive kernel
       Nanalyse - number of points to analyse in one bach. Default is 2**16 points.
       k        - number of min neighbors for the adaptive kernel smoothing. If k=0 : fixed kernel.
       

   Output :

       res - list of two dimensional numpy matrix (of size length(data) X length(radius)

              res[Ø] = Wmean   : weighted mean
              res[1] = Wstd    : weighted standart deviation
              res[2] = NWratio : non weighted mean
              res[3] = NWstd   : non weighted standart deviation
              res[4] = Mom     : weighted moment (order 0 to 4) of the absolute value
                      of thecoefficient. matrix of size 5 X length(data) X length(radius)
              res[5] = Cum     : weighted cumulant one and two 
                      of thecoefficient. matrix of size 2 X length(data) X length(radius)



    ##
     S.G.  Roux, ENS Lyon, December 2020,  stephane.roux@ens-lyon.fr
     J.L June 2021
     S.G.  Roux, ENS Lyon, November 2021
    """
    assert(Count1.shape == Count2.shape)
    
    si=data.shape
    if si[1] < 2 or si[1]>3:
        raise TypeError('The second dimension of first argument must be of length 2 or 3.')

    if Count1.ndim > 2:
        raise TypeError('The second arument must be two or threedimensional.')
   
    if Count1.shape[0] != si[0]:
        raise TypeError('The two first arument must have the same length.')

    if radius[radius>T].shape[0] >0:
        raise TypeError('The last argument must be greater than a sqrt(2)*radius.')
    # check grid
    if np.sum(np.abs(np.array(X.shape)-np.array(Y.shape)))>0:
        raise TypeError('X and Y must have the same size.')

    if np.sum(np.diff(X,axis=1,n=2))>0:
        raise TypeError('X must be regulary sampled.')
    else:
        gridsizex=max(X[0][1] - X[0][0],X[1][0] - X[0][0])
        
    if np.sum(np.diff(Y,axis=0,n=2))>0:
        raise TypeError('Y must be regulary sampled.')
    else:
        gridsizey=max(Y[1][0] - Y[0][0],Y[0][1] - Y[0][0])
    
    if gridsizex != gridsizey:
        raise TypeError('X and Y must have same sampling.')
    
    if radius.size != Count1.shape[1]:
        raise TypeError('The size of the second arument must be [N,R,M] where R is the number of scales (=radius.size)')
 
    # create grid points
    mygrid=np.column_stack([X.flatten(),Y.flatten()])
    gridlength=mygrid.shape[0]
    gridsizecurr = gridsizex

    # how many variable and number of pairs 
    #Nvar=Count.shape[2]
    #allpairs=list(combinations(range(Nvar),2))     
    #Npairs=len(allpairs)
    #Nvar=2
    Npairs=1
    print('There is {:d} different pair(s)'.format(Npairs))
    
    Nanalyse2=300000
    if Nanalyse==0:
        print('Warning : dangerous if your set of data is large.')
        Nanalyse=gridlength
        Nbunch=1
    else: # find how many bunchs        
        Nbunch = int(np.ceil(gridlength/Nanalyse))
    
    # choose weighting  
    if isinstance(weights, str):
        if weights == 'flat':
            weightingfunction = partial(geographicalWeight,func = flatWindow)
            print('No weight')
        elif weights == 'Epanechnikov':
            weightingfunction = partial(geographicalWeight,func = EpanechnikovWindow)
        elif weights == 'bisquare':
            weightingfunction = partial(geographicalWeight,func = bisquareWindow)
        elif weights == 'triangular':
            weightingfunction = partial(geographicalWeight,func = triangularWindow)
        elif weights == 'tricube':
            weightingfunction = partial(geographicalWeight,func = tricubeWindow)            
    elif isinstance(weights, types.FunctionType):
        weightingfunction = partial(geographicalWeight,func = weights)
    else:
        raise TypeError('ERROR : weighting is of unknown type.).')

    if adaptive == True:
        if len(Tloc) ==0: # compute local env
            print('ADAPTIVE KERNEL: Computing pilot density')
            CountT= localWaveTrans(data[:,(0,1)], np.atleast_1d(T))
            #out_T = LmsAnalysis(data[:,(0,1)],CountT,X,Y,np.atleast_1d(T), T=T,Nanalyse=2**9)
            #Wmean_T, Wstd_T, NWmean_T, NWstd_T, Mom_T, Cum_T = out_T
            WmeanStd_T, Mom_T, Flat_T, Cum_T = LmsAnalysis(data[:,(0,1)],CountT,X,Y,np.atleast_1d(T), T=T,Nanalyse=2**9)
            Wmean_Tloc = WmeanStd_T[0,:,0]
            G = np.nanmean(WmeanStd_T[0,:,0])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                lambda_ie = G/Wmean_Tloc
            Tloc = lambda_ie*T
            Tlocmax = np.nanquantile(Tloc,0.8)
            Tloc[Tloc>Tlocmax] = Tlocmax
            
            T=Tlocmax
            print('ADAPTIVE KERNEL computed.')
        elif X.flatten().shape[0] == Tloc.flatten().shape[0]:
            T=np.nanmax(Tloc)
        else: 
            raise TypeError('ERROR : wrong size for the local environment')
        #Tloc=Tloc*0+200
    else:
        Tloc=T*np.ones(X.size)

    # results allocations
    rho = np.nan*np.zeros( (gridlength,radius.size), dtype = float)
       
    # loop on the bunch
    for ibunch in range(Nbunch):
        # %
        print('bunch {:d}/{:d} '.format(ibunch+1,Nbunch), end=' ')
        sys.stdout.flush()
        # get data index of current bunch
        index=np.arange(ibunch*Nanalyse,(ibunch+1)*Nanalyse,1)
        index=index[index<gridlength]

        # we restrict the tree to points whithin a radius T 
        mi=np.min(mygrid[index,:], axis=0)
        ma=np.max(mygrid[index,:], axis=0)
        IndexNear=np.where((data[:,0] >mi[0]-T) & (data[:,0] <ma[0]+T) & (data[:,1]  >mi[1]-T) & (data[:,1] <ma[1]+T))[0]

        # make the tree with the nearest points only
        if IndexNear.shape[0]>0:
            tree = KDTree(data[IndexNear,0:2])
            Idxtot = tree.query_radius(mygrid[index,:], r = np.sqrt(2)*gridsizecurr, count_only=False, return_distance=False)
        
        
            # find and remove empty (no point location) grid pixels
            thelengths=np.array([ Idxtot[igrid].shape[0] for igrid in range (len(Idxtot))])
            IdxMin, = np.where(thelengths>0.)
            index = index[IdxMin]
            Idxtot=Idxtot[IdxMin]
            thelengths=thelengths[IdxMin]
        
            # Managed the number of grid points  (too many neighboors for some)
            cumsumbunch=np.cumsum(thelengths)
            Nflowers=int(np.ceil(np.sum(thelengths)/Nanalyse2))
            print('{:d} batch(s).'.format(Nflowers))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                for iflower in range(Nflowers):
                    i1,=np.where(iflower*Nanalyse2 <= cumsumbunch)
                    i2,=np.where(cumsumbunch <  (iflower+1)*Nanalyse2)
                    flowers=np.intersect1d(i1,i2)
                    rho[index[flowers],:] = computeGridValuebivar(tree,Count1,Count2,IndexNear,index[flowers],mygrid, radius,Tloc[index[flowers]],weightingfunction)
                    #tt = computeGridValuebivar(tree,Count1,Count2,IndexNear,index[flowers],mygrid, radius,Tloc[index[flowers]],weightingfunction)
                    #print(tt.shape,rho.shape)
        print('.')   
       
    return rho

#%%
#%%
if __name__ == '__main__':
    # test signal conv
    N=512
    data = np.random.randn(N,)
    radius = np.array([20, 40, 80 ])
    L=200
    # Hat and poor
    Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = MultiScaleQuantityConv1d(data, radius,wavelet ='harr')
    Mean_Std_loc, Mom12_loc, Flatness , Cum12_loc = localMomCumConv1d(Poor,LPoor,L,window='flat')
        
    
    
    #%%
    
    #print('ici ', Mom2_loc.shape)
    maingrrr                   
    # Harr and rigthpoor 
    Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = MultiScaleQuantityConv1d(data, radius,wavelet ='harr')
    
    # test image conv
    N=512
    data=np.random.randn(N,N)
    radius = np.array([20, 40, 80, 160])
    L = 200
    Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd  = MultiScaleQuantityConv(data, radius)
    Mean_Std, Mom12, Flatness ,Cum12 = localMomCumConv2d(Poor,LPoor,L, window='circle',fftplan=[], gridsize=1)
    
    # test non marked point process
    data = np.random.randn(N, 2)
    radius = np.array([.01, .2, .4, .8, 1.6])

    Count, HatSupport = MultiScaleQuantitySupportKnn(data,radius)
    
    # test non marked point process   
    data = np.random.randn(N, 3)
    radius = np.array([.01, .2, .4, .8, 1.6])
    L = 2
    # WARNING : PB with destination because no value at these points  : TO CORRECT
    Count, HatSupport, Mean, LMean, HatMean, LHatMean, Poor, LPoor, Std, HatStd = MultiScaleQuantityMarkKnn(
        data, radius)

    gridsize = .5
    y = np.arange(np.min(data[:, 0])-gridsize/2,
                  np.max(data[:, 0])+gridsize/2, gridsize)
    x = np.arange(np.min(data[:, 1])-gridsize/2,
                  np.max(data[:, 1])+gridsize/2, gridsize)

    X, Y = np.meshgrid(x, y)
    Mean_Std, Mom12, Flatness, Cum12 = LmsAnalysis(
        data[:, 0:2], Count, Y, X, radius, T=L, adaptive=True)

    #%%
    rho = localCorrCoefKnn( data[:, 0:2], LMean, LMean, Y, X, radius, T=L, adaptive=True)
    
    plt.clf()
    plt.imshow(rho[:,0].reshape(X.shape))
    plt.colorbar()