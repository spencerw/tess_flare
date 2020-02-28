import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits
from flareTools import id_segments

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

prefix = 'sec1_small'
path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/' + prefix + '/'
df = pd.read_csv(prefix+'_flare_out.csv')
df_par = pd.read_csv(prefix+'_param_out.csv')

files = np.unique(df['file'].values)

for filename in files:
    tstart = df[df['file'] == filename]['t0'].values
    tstop = df[df['file'] == filename]['t1'].values
    
    gauss_fit = False
    entry = df[df['file'] == filename]
    if entry.iloc[0]['g_chisq'] < df.iloc[0]['f_chisq']:
        gauss_fit = True
    
    fig, axes = plt.subplots(figsize=(16,8), nrows=1, ncols=2)
    
    with fits.open(path+filename, mode='readonly') as hdulist:
        tess_bjd = hdulist[1].data['TIME']
        pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
        pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']
        quality = hdulist[1].data['QUALITY']
        
    time_smo, smo = np.loadtxt(path+filename+'.gp')
    ok_cut = (quality == 0) & (~np.isnan(tess_bjd)) & (~np.isnan(pdcsap_flux)) & (~np.isnan(pdcsap_flux_error))
    
    dt_limit = 12/24 # 12 hours
    trim = 4/24 # 4 hours
    istart, istop = id_segments(tess_bjd[ok_cut], dt_limit, dt_trim=trim)

    time_c = np.array([])
    flux_c = np.array([])
    error_c = np.array([])

    for seg_idx in range(len(istart)):
        tess_bjd_seg = tess_bjd[ok_cut][istart[seg_idx]:istop[seg_idx]]
        pdcsap_flux_seg = pdcsap_flux[ok_cut][istart[seg_idx]:istop[seg_idx]]
        pdcsap_flux_error_seg = pdcsap_flux_error[ok_cut][istart[seg_idx]:istop[seg_idx]]

        time_c = np.concatenate((time_c, tess_bjd_seg), axis=0)
        flux_c = np.concatenate((flux_c, pdcsap_flux_seg), axis=0)
        error_c = np.concatenate((error_c, pdcsap_flux_error_seg), axis=0)
        
    median = np.nanmedian(flux_c)
    
    axes[0].plot(time_c, flux_c/median)
    axes[0].plot(time_smo, smo)
    axes[0].set_xlabel('Time [BJD - 2457000, days]')
    axes[0].set_ylabel('Flux [e-/s]')
    axes[0].set_title(filename)
    
    x = time_c
    y = flux_c/median - smo + 1
    axes[1].plot(x, y)
    for idx in range(len((tstart))):
        indices = np.where((x >= tstart[idx]) & (x <= tstop[idx]))[0]
        marker = 'o'
        if gauss_fit:
            marker = 'x'
        axes[1].plot(x[indices], y[indices], marker)
    axes[1].set_xlabel('Time [BJD - 2457000, days]')
    axes[1].set_ylabel('Normalized Flux')
    plt.savefig('/astro/store/gradscratch/tmp/scw7/tessData/plots/'+filename+'.png')
    plt.close()
