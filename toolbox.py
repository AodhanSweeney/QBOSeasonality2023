import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from scipy import stats

def starter():
    """
    The starter() function loads processed data of temperature (from GPS-RO measurements),
    TTL cirrus cloud fraction (from CALIPSO), and zonal wind data from ERA5. Temperature and
    cloud fraction data have 0.1 km vertical resolution from 12 to 22 km. Zonal wind data is
    output on 100 linearly interpolated pressure levels from 50 to 350 hPa. All data has been
    gridded to 2.5x2.5 horizontal resolution for the 15 year period from 2006-2020.
    """

    # base_path tells you where to look for the data
    base_path = '/home/disk/p/aodhan/' 
    
    # Pull in Temperature data
    temp_map_prof_files = glob.glob(base_path + 'cf_physical_parameters_correlations/tempmaps/TempMapsC1MAMB/*Prof*.npy')
    temp_map_prof_files = np.sort(temp_map_prof_files)
    temp_profs = [np.load(temp_map_prof_files[yr]) for yr in range(len(temp_map_prof_files))]

    # TTL Cirrus CF data
    profile_cf_map_files_old_old = glob.glob(base_path + 'cf_physical_parameters_correlations/aerosol_cloud_distinctions/cfmaps/TTLcfMonthlyProfiles_strataerosolremoved_*.npy')
    profile_cf_map_files_old = []
    for file in profile_cf_map_files_old_old:
        if len(file) == len(base_path + 'cf_physical_parameters_correlations/aerosol_cloud_distinctions/cfmaps/TTLcfMonthlyProfiles_strataerosolremoved_2006_06.npy'):
            profile_cf_map_files_old.append(file)
    profile_cf_map_files_ = np.sort(profile_cf_map_files_old)[:-8]
    profile_cf = np.array([np.load(profile_cf_map_files_[yr])[0] for yr in range(len(profile_cf_map_files_))])
    empty_prof_map = np.empty(np.shape(profile_cf[:5]))
    empty_prof_map[:] = np.NaN
    profile_cf = np.concatenate((empty_prof_map, profile_cf), axis=0)
    profile_TTLCCF = np.reshape(profile_cf, (15,12,24,144,101))
     # average over missing CALIPSO data (feb 2016)
    profile_TTLCCF[10,1] = np.nanmean([profile_TTLCCF[10,0], profile_TTLCCF[10,2]], axis=0)

    # All other cloud fraction data
    all_cf_prof_maps= glob.glob(base_path + 'cf_physical_parameters_correlations/aerosol_cloud_distinctions/cfmaps/ALLcfMonthlyProfiles_strataerosolremoved_1*.npy')
    opaque_cf_prof_maps = glob.glob('/home/bdc/aodhan/CFmaps/TTLcfMonthlyProfiles_strataerosolremoved_*opaque.npy')
    transparent_cf_prof_maps = glob.glob('/home/bdc/aodhan/CFmaps/TTLcfMonthlyProfiles_strataerosolremoved_*_transparent_noTTLcirrus.npy')
    profile_ALLCF = cf_profile_finder(all_cf_prof_maps)
    profile_TRANS = cf_profile_finder(transparent_cf_prof_maps)
    profile_OPAQUE = cf_profile_finder(opaque_cf_prof_maps)

    #zonal wind
    zonal_wind = np.load('/usb/zonalwindERA5/plevel_zonal_winds.npy')
    zonal_wind = np.reshape(zonal_wind, (15,12,24,144,100))

    return(temp_profs, profile_TTLCCF, profile_ALLCF, profile_TRANS, profile_OPAQUE, zonal_wind)

def cf_profile_finder(cf_prof_maps):
    """
    The cf_profile_finder() function loads specific cloud fraction data (other than TTL Cirrus clouds).
    Data returned is the zonal mean cloud fraction based on cloud type (opaque, transparent, or all) 
    on a 2.5 latitude grid from 0 to 22 km with 0.1 km spacing.
    """

    cf_prof_maps_ = np.sort(cf_prof_maps)
    profile_cf = np.array([np.load(cf_prof_maps_[yr])[0] for yr in range(len(cf_prof_maps_))])
    empty_prof_map = np.empty(np.shape(profile_cf[:5]))
    empty_prof_map[:] = np.NaN
    profile_cf = np.concatenate((empty_prof_map, profile_cf), axis=0)
    profile_cf = np.reshape(profile_cf, (15,12,24,144,221))
    # average over missing CALIPSO data (feb 2016)
    profile_cf[10,1] = np.nanmean([profile_cf[10,0], profile_cf[10,2]], axis=0)
    profile_cf_anoms = profile_cf - np.nanmean(profile_cf, axis=0)
    profile_cf_anoms_zm = np.nanmean(profile_cf_anoms, axis=3)
    return(profile_cf_anoms_zm)


def anomaly_finder(data_calendar):
    """ 
    The anomaly_finder() removes the seasonal cycle from the data. 
    Data should be indexed as (years, months, lats, lons)
    """

    seasonal_average = np.nanmean(data_calendar, axis=0)
    anomalies_calendar = data_calendar - seasonal_average
    return anomalies_calendar

def vert_temp_gradient(data_calendar):
    """
    The vert_temp_gradient() takes derivative with respect to height of monthly mean
    temperature profiles. 
    """

    data_calendar = np.array(data_calendar)

    # create data with different start and end points
    t_1 = data_calendar[:,:,:,:,1:]
    t_2 = data_calendar[:,:,:,:,:-1]

    # find derivative and average
    dt_dz = (t_1 - t_2)/0.1 # K/km
    dt_dz_1 = dt_dz[:,:,:,:,1:]
    dt_dz_2 = dt_dz[:,:,:,:,:-1]
    dt_dz_new = np.nanmean([dt_dz_1, dt_dz_2], axis=0)
    empty = np.empty(np.shape(data_calendar[:,:,:,:,0]))
    empty[:] = np.NaN
    dt_dz_new = np.insert(dt_dz_new, 98, empty, axis=4)
    dt_dz_new = np.insert(dt_dz_new, 0, empty, axis=4)
    return(dt_dz_new)

def get_eq_mean(calendar):
    """
    get_eq_mean() takes the mean from 10S-10N over an array given that the data's 
    latitude axis is axis=1 and that the data is spaced from 30S-30N with 2.5 lat spacing
    """

    eq_mean_calendar = np.nanmean(calendar[:,8:16,:], axis=1)    
    return(eq_mean_calendar)

def alt2pres(altitude):
    H = 7 
    press = 1000*np.exp(-1*(altitude/H))
    return press

def press2alt(press):
    H = 7 
    altitude = -1*H*np.log(press/1000)
    return altitude

