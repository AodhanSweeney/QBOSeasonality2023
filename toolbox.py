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

def season_finder_profiles(calendar, m1, m2, m3,):
    """ 
    Data should be indexed as (years, months)
    """
    total_djf_anoms_list = [calendar[:,m1], calendar[:,m2], 
                            calendar[:,m3]]
    total_djf_anoms = np.swapaxes(total_djf_anoms_list, 0,1)
    shape = np.shape(total_djf_anoms)
    total_djf_timeseries = np.reshape(total_djf_anoms, (shape[0]*shape[1],shape[2], shape[3]))
    return(total_djf_timeseries)

def season_finder(calendar, m1, m2, m3):
    """ 
    Data should be indexed as (years, months)
    """
    total_djf_anoms_list = [calendar[:,m1], calendar[:,m2], 
                            calendar[:,m3]]
    total_djf_anoms = np.swapaxes(total_djf_anoms_list, 0,1)
    shape = np.shape(total_djf_anoms)
    total_djf_timeseries = np.reshape(total_djf_anoms, (shape[0]*shape[1]))
    return(total_djf_timeseries)

def regression_ts(pc, tmap):    
    x_indices = np.shape(tmap)[1]
    y_indices = np.shape(tmap)[2]
    r_map = []
    for x in range(x_indices):
        r_x = []
        for y in range(y_indices):
            temp_series = tmap[:,x,y]
            temp_series_no_nan = temp_series[~np.isnan(temp_series)]
            pc_no_nan = pc[~np.isnan(temp_series)]
            try:
                reg = stats.linregress(pc_no_nan, temp_series_no_nan)
                reg_ts = reg[1] + reg[0]*pc
            except:
                reg_ts = np.repeat(np.NaN, len(temp_series))
            r_x.append(reg_ts)
        r_map.append(r_x)
    r_map = np.transpose(r_map)
    return(np.swapaxes(r_map, 1,2))

def cpt_alt_finder(temp_profs):
    calendar = []
    for yr in range(15):
        year_map = []
        for mon in range(12):
            month_map = []
            for lat in range(24):
                lat_line = []
                for lon in range(144):
                    try:
                        cpt_alt = np.nanargmin(temp_profs[yr,mon,lat,lon])/10 + 12
                    except:
                        cpt_alt = np.NaN
                    lat_line.append(cpt_alt)
                month_map.append(lat_line)
            year_map.append(month_map)
        calendar.append(year_map)
    calendar = np.array(calendar)
    return(calendar)

def difference_and_significance_map(qbo_season, variable_eq_season):
    w_variable_season = variable_eq_season[qbo_season > .5]
    e_variable_season = variable_eq_season[qbo_season < -.5]
    w_m_e_season = np.nanmean(w_variable_season, axis=0) - np.nanmean(e_variable_season, axis=0)
    w_season = np.nanmean(w_variable_season, axis=0)
    e_season = np.nanmean(e_variable_season, axis=0)
    difference_distribution = []
    w_distribution = []
    e_distribution = []
    for boot_strap_idx in range(1000):
        rand_indices_w = np.random.randint(low=0, high=len(w_variable_season), size=len(w_variable_season))
        random_w_cf_seasons = w_variable_season[rand_indices_w]
        w_distribution.append(np.nanmean(random_w_cf_seasons, axis=0))
        rand_indices_e = np.random.randint(low=0, high=len(e_variable_season), size=len(e_variable_season))
        random_e_cf_seasons = e_variable_season[rand_indices_e]
        e_distribution.append(np.nanmean(random_e_cf_seasons, axis=0))
        random_difference = np.nanmean(random_w_cf_seasons, axis=0) - np.nanmean(random_e_cf_seasons, axis=0)
        difference_distribution.append(random_difference)
    two_sigma_significance = np.nanstd(difference_distribution, axis=0)*2
    two_sigma_w_significance = np.nanstd(w_distribution, axis=0)*2
    two_sigma_e_significance = np.nanstd(e_distribution, axis=0)*2
    return(w_m_e_season, two_sigma_significance, w_season, 
           two_sigma_w_significance, e_season, two_sigma_e_significance)

def three_month_smoother(temp_profile_anoms_cal_west):    
        
    smoothed_west_anoms_DJF = [temp_profile_anoms_cal_west[:,11], temp_profile_anoms_cal_west[:,0], temp_profile_anoms_cal_west[:,1]]
    smoothed_west_anoms_JFM = [temp_profile_anoms_cal_west[:,0], temp_profile_anoms_cal_west[:,1], temp_profile_anoms_cal_west[:,2]]
    smoothed_west_anoms_FMA = [temp_profile_anoms_cal_west[:,1], temp_profile_anoms_cal_west[:,2], temp_profile_anoms_cal_west[:,3]]
    smoothed_west_anoms_MAM = [temp_profile_anoms_cal_west[:,2], temp_profile_anoms_cal_west[:,3], temp_profile_anoms_cal_west[:,4]]
    smoothed_west_anoms_AMJ = [temp_profile_anoms_cal_west[:,3], temp_profile_anoms_cal_west[:,4], temp_profile_anoms_cal_west[:,5]]
    smoothed_west_anoms_MJJ = [temp_profile_anoms_cal_west[:,4], temp_profile_anoms_cal_west[:,5], temp_profile_anoms_cal_west[:,6]]
    smoothed_west_anoms_JJA = [temp_profile_anoms_cal_west[:,5], temp_profile_anoms_cal_west[:,6], temp_profile_anoms_cal_west[:,7]]
    smoothed_west_anoms_JAS = [temp_profile_anoms_cal_west[:,6], temp_profile_anoms_cal_west[:,7], temp_profile_anoms_cal_west[:,8]]
    smoothed_west_anoms_ASO = [temp_profile_anoms_cal_west[:,7], temp_profile_anoms_cal_west[:,8], temp_profile_anoms_cal_west[:,9]]
    smoothed_west_anoms_SON = [temp_profile_anoms_cal_west[:,8], temp_profile_anoms_cal_west[:,9], temp_profile_anoms_cal_west[:,10]]
    smoothed_west_anoms_OND = [temp_profile_anoms_cal_west[:,9], temp_profile_anoms_cal_west[:,10], temp_profile_anoms_cal_west[:,11]]
    smoothed_west_anoms_NDJ = [temp_profile_anoms_cal_west[:,10], temp_profile_anoms_cal_west[:,11], temp_profile_anoms_cal_west[:,0]]

    smoothed_temp_profile_anoms_cal_west = [smoothed_west_anoms_DJF, smoothed_west_anoms_JFM,
                                            smoothed_west_anoms_FMA, smoothed_west_anoms_MAM,
                                            smoothed_west_anoms_AMJ, smoothed_west_anoms_MJJ,
                                            smoothed_west_anoms_JJA, smoothed_west_anoms_JAS,
                                            smoothed_west_anoms_ASO, smoothed_west_anoms_SON,
                                            smoothed_west_anoms_OND, smoothed_west_anoms_NDJ]
    
    smoothed_temp_profile_anoms_cal_west_ = np.nanmean(smoothed_temp_profile_anoms_cal_west, axis=1)
    smoothed_temp_profile_anoms_cal_west_ = np.nanmean(smoothed_temp_profile_anoms_cal_west_, axis=1)
    smoothed_temp_profile_anoms_cal_west_ = np.insert(smoothed_temp_profile_anoms_cal_west_, 12, smoothed_temp_profile_anoms_cal_west_[0])
    return(smoothed_temp_profile_anoms_cal_west_)

def lead_impact_finder(qbo_file, data_calendar):
    enso_index = np.load('/home/disk/p/aodhan/large_scale_dynamics/Monthly_ERSSTv5_Niño_3p4_1979_2020.npy')[-180:]
    data_roladex = []
    seasons = [[11,0,1], [0,1,2], [1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
                [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,0]]
    for lead in range(5):
        lead_data = []
        if lead == 0:
            qbo_index = np.load(qbo_file)
            qbo = qbo_index - np.nanmean(qbo_index)
            qbo_index = qbo/np.nanstd(qbo)
            qbo_index = qbo_index[-180:]
        else:
            start = -180 - lead
            end = -1*lead
            qbo_index = np.load(qbo_file)
            qbo = qbo_index - np.nanmean(qbo_index)
            qbo_index = qbo/np.nanstd(qbo)
            qbo_index = qbo_index[start:end]
        for season in seasons:
            data_season = season_finder_profiles(data_calendar, season[0], season[1], season[2])[:]
            qbo_season = season_finder(np.reshape(qbo_index, (15,12)), season[0], season[1], season[2])[:]
            ###################### Remove ENSO 3.4 signal ########################################
            enso_season = season_finder(np.reshape(enso_index, (15,12)), season[0], season[1], season[2])
            data_season = data_season - regression_ts(enso_season, data_season)
            ####################################################################################
            data_eq_season = get_eq_mean(data_season)
            data_WmE_composite = difference_and_significance_map(qbo_season, data_eq_season)
            lead_data.append(data_WmE_composite)
        data_roladex.append(lead_data)
    
    data_roladex = np.array(data_roladex)
    return data_roladex