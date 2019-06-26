import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
import pandas as pd
import numpy as np
import exoplanet as xo
import os

from flareTools import FINDflare, IRLSSpline, update_progress

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

def procFlares(files, sector, makefig=True, clobber=False):

    FL_id = np.array([])
    FL_t0 = np.array([])
    FL_t1 = np.array([])
    FL_f0 = np.array([])
    FL_f1 = np.array([])

    for k in range(len(files)):
        filename = files[k].split('/')[-1]
        acf_1dt = 0

        # Would be nice if I could open this with astropy.Table.read
        # and turn it into a pandas dataframe
        # When I try, it tells me 'IOError: Header missing END card.'
        with fits.open(files[k], mode='readonly') as hdulist:
            tess_bjd = hdulist[1].data['TIME']
            quality = hdulist[1].data['QUALITY']
            pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
            pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']
            
        ok_cut = (quality == 0)
        
        median = np.nanmedian(pdcsap_flux[ok_cut])
        acf = xo.autocorr_estimator(tess_bjd[ok_cut], pdcsap_flux[ok_cut]/median,
                                    yerr=pdcsap_flux_error[ok_cut]/median,
                                    min_period=0.1, max_period=27, max_peaks=2)
        
        if len(acf['peaks']) > 0:
            acf_1dt = acf['peaks'][0]['period']
            mask = np.where((acf['autocorr'][0] == acf['peaks'][0]['period']))[0]
            acf_1pk = acf['autocorr'][1][mask][0]
            s_window = int(acf_1dt/np.fabs(np.nanmedian(np.diff(tess_bjd))) / 6)
        else:
            s_window = 128
        
        # This is annoying, since I can't create a dataframe from the FITS data, I have
        # to do all of this extra crap
        smo = pd.DataFrame(pdcsap_flux[ok_cut]).rolling(s_window, center=True).median().values
        smo = smo.reshape(1, -1)[0]

        if makefig:
            fig, axes = plt.subplots(figsize=(8,8))
            axes.errorbar(tess_bjd[ok_cut], pdcsap_flux[ok_cut]/median,
                          yerr=pdcsap_flux_error[ok_cut]/median, 
                          linestyle=None, alpha=0.15, label='PDCSAP_FLUX')
            axes.plot(tess_bjd[ok_cut], smo/median, label=str(s_window)+'pt median')

            if (acf_1dt > 0):
                y = np.nanstd(smo/median)*acf_1pk*np.sin(tess_bjd[ok_cut]/acf_1dt*2*np.pi) + 1
                axes.plot(tess_bjd[ok_cut], y, lw=2, alpha=0.7,
                         label='ACF='+format(acf_1dt,'6.3f')+'d, pk='+format(acf_1pk,'6.3f'))
            
        if np.sum(ok_cut) < 1000:
            print('Warning: ' + f + ' contains < 1000 good points')
            
        sok_cut = np.isfinite(smo)
        
        FL = FINDflare(pdcsap_flux[ok_cut][sok_cut] - smo[sok_cut]/median, 
                       pdcsap_flux_error[ok_cut][sok_cut]/median,
                       N1=4, N2=2, N3=5, avg_std=False)
        
        for j in range(len(FL[0])):
            FL_id = np.append(FL_id, k)
            FL_t0 = np.append(FL_t0, FL[0][j])
            FL_t1 = np.append(FL_t1, FL[1][j])
            FL_f0 = np.append(FL_f0, median)
            s1, s2 = FL[0][j], FL[1][j]+1
            FL_f1 = np.append(FL_f1, np.nanmax(pdcsap_flux[ok_cut][sok_cut][s1:s2]))
            
            if makefig:
                axes.scatter(tess_bjd[ok_cut][sok_cut][s1:s2],
                             pdcsap_flux[ok_cut][sok_cut][s1:s2]/median,
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
        
        
        update_progress(k / len(files))
        
    ALL_TIC = pd.Series(files).str.split('-', expand=True).iloc[:,-3].astype('int')
    print(str(len(ALL_TIC[FL_id])) + ' flares found across ' + str(len(files)) + ' files')
    flare_out = pd.DataFrame(data={'TIC':ALL_TIC[FL_id], 'i0':FL_t0, 'i1':FL_t1,
                                   'med':FL_f0, 'peak':FL_f1})
    flare_out.to_csv(sector+ '_flare_out.csv')
