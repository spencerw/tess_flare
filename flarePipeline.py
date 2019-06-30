import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import exoplanet as xo
import os

from flareTools import FINDflare, IRLSSpline, update_progress

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

def procFlares(files, sector, makefig=True, clobber=False, doSpline=False, progCounter=False):

    FL_id = np.array([])
    FL_t0 = np.array([])
    FL_t1 = np.array([])
    FL_f0 = np.array([])
    FL_f1 = np.array([])

    for k in range(len(files)):
        filename = files[k].split('/')[-1]
        acf_1dt = 0

        with fits.open(files[k], mode='readonly') as hdulist:
            tess_bjd = hdulist[1].data['TIME']
            quality = hdulist[1].data['QUALITY']
            pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
            pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']
            
            tbl = Table([tess_bjd, quality, pdcsap_flux, pdcsap_flux_error], 
                        names=('TIME', 'QUALITY', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'))
            df_tbl = tbl.to_pandas()
            
        ok_cut = (tbl['QUALITY'] == 0)
        
        median = np.nanmedian(df_tbl['PDCSAP_FLUX'][ok_cut])
        
        if doSpline:
            smo = IRLSSpline(df_tbl['TIME'].values[ok_cut], df_tbl['PDCSAP_FLUX'].values[ok_cut]/median,
                         df_tbl['PDCSAP_FLUX_ERR'].values[ok_cut]/median)
        else:
            acf = xo.autocorr_estimator(tbl['TIME'][ok_cut], tbl['PDCSAP_FLUX'][ok_cut]/median,
                                        yerr=tbl['PDCSAP_FLUX_ERR'][ok_cut]/median,
                                        min_period=0.1, max_period=27, max_peaks=2)

            if len(acf['peaks']) > 0:
                acf_1dt = acf['peaks'][0]['period']
                mask = np.where((acf['autocorr'][0] == acf['peaks'][0]['period']))[0]
                acf_1pk = acf['autocorr'][1][mask][0]
                s_window = int(acf_1dt/np.fabs(np.nanmedian(np.diff(df_tbl['TIME']))) / 6)
            else:
                s_window = 128

            smo = df_tbl['PDCSAP_FLUX'][ok_cut].rolling(s_window, center=True).median()
        

        if makefig:
            fig, axes = plt.subplots(figsize=(8,8))
            axes.errorbar(df_tbl['TIME'][ok_cut], df_tbl['PDCSAP_FLUX'][ok_cut]/median,
                          yerr=df_tbl['PDCSAP_FLUX_ERR'][ok_cut]/median, 
                          linestyle=None, alpha=0.15, label='PDCSAP_FLUX')
            axes.plot(df_tbl['TIME'][ok_cut], smo/median, label=str(s_window)+'pt median')

            if (acf_1dt > 0):
                y = np.nanstd(smo/median)*acf_1pk*np.sin(df_tbl['TIME'][ok_cut]/acf_1dt*2*np.pi) + 1
                axes.plot(df_tbl['TIME'][ok_cut], y, lw=2, alpha=0.7,
                         label='ACF='+format(acf_1dt,'6.3f')+'d, pk='+format(acf_1pk,'6.3f'))
            
        if np.sum(ok_cut) < 1000:
            print('Warning: ' + f + ' contains < 1000 good points')
            
        sok_cut = np.isfinite(smo)
        
        FL = FINDflare((df_tbl['PDCSAP_FLUX'][ok_cut][sok_cut] - smo[sok_cut])/median, 
                       df_tbl['PDCSAP_FLUX_ERR'][ok_cut][sok_cut]/median,
                       avg_std=False, N1=4, N2=2, N3=5)
        
        for j in range(len(FL[0])):
            FL_id = np.append(FL_id, k)
            FL_t0 = np.append(FL_t0, df_tbl['TIME'][ok_cut][sok_cut].values[FL[0][j]])
            FL_t1 = np.append(FL_t1, df_tbl['TIME'][ok_cut][sok_cut].values[FL[1][j]])
            FL_f0 = np.append(FL_f0, median)
            s1, s2 = FL[0][j], FL[1][j]+1
            FL_f1 = np.append(FL_f1, np.nanmax(df_tbl['PDCSAP_FLUX'][ok_cut][sok_cut][s1:s2]))
            
            if makefig:
                axes.scatter(df_tbl['TIME'][ok_cut][sok_cut][s1:s2],
                             df_tbl['PDCSAP_FLUX'][ok_cut][sok_cut][s1:s2]/median,
                             color='r', label='_nolegend_')
                axes.scatter([],[], color='r', label='Flare?')        

        if makefig:
            axes.set_xlabel('Time [BJD - 2457000, days]')
            axes.set_ylabel('Normalized Flux')
            axes.legend()
            axes.set_title(filename)

            figname = filename + '.jpeg'
            makefig = ((not os.path.exists(figname)) | clobber)
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.25, dpi=100)
            plt.close()      
        
        if progCounter:
            update_progress(k / len(files))
        
    ALL_TIC = pd.Series(files).str.split('-', expand=True).iloc[:,-3].astype('int')
    print(str(len(ALL_TIC[FL_id])) + ' flares found across ' + str(len(files)) + ' files')
    flare_out = pd.DataFrame(data={'TIC':ALL_TIC[FL_id], 'i0':FL_i0, 'i1':FL_i1,
                                   't0':FL_t0, 't1':FL_t1,
                                   'med':FL_f0, 'peak':FL_f1})
    flare_out.to_csv(sector+ '_flare_out.csv')