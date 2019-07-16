import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
import pandas as pd
import numpy as np
import exoplanet as xo
import os
import celerite
from celerite import terms
from scipy.optimize import minimize

from flareTools import FINDflare, IRLSSpline, id_segments, update_progress

mpl.rcParams.update({'font.size': 18, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                            'image.cmap': 'viridis'})

def iterGaussProc(time, flux, flux_err, period_guess, interval=10, num_iter=5):
    
    x = time[::interval]
    y = flux[::interval]
    yerr = flux_err[::interval]
    
    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    bounds = dict(log_S0=(-20, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                           bounds=bounds)
    kernel.freeze_parameter("log_Q")  # We don't want to fit for "Q" in this term

    # A periodic component
    Q = 1.0
    w0 = 2*np.pi/period_guess
    S0 = np.var(y) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                            bounds=bounds)
    
    kernel += terms.JitterTerm(log_sigma=np.log(np.median(yerr)),
                               bounds=[(-20.0, 5.0)])

    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(x, yerr)  # You always need to call compute once.

    def neg_log_like(params, y, gp, m):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y[m])

    def grad_neg_log_like(params, y, gp, m):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y[m])[1]

    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    m = np.ones(len(x), dtype=bool)
    for i in range(num_iter):
        gp.compute(x[m], yerr[m])
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method="L-BFGS-B", bounds=bounds, args=(y, gp, m))
        gp.set_parameter_vector(soln.x)
    #     print(soln)
        initial_params = soln.x
        # log_a, logQ1, mix_par, logQ2, log_P
    #     print(initial_params)
        mu, var = gp.predict(y[m], x, return_var=True)
        sig = np.sqrt(var + yerr**2)

        m0 = y - mu < 1.3 * sig
        #print(m0.sum(), m.sum())
        if np.all(m0 == m) or (np.abs(m0.sum()- m.sum()) < 3)  or (m0.sum() == 0):
            break
        m = m0

    fit_x, fit_y, fit_yerr = x[m], y[m], yerr[m]

    gp.compute(fit_x, fit_yerr)
    gp.log_likelihood(fit_y)   
    gp.get_parameter_dict()
    
    # We want mu and var to be the same shape as the time array, need to interpolate
    # since we downsampled
    mu_interp = np.interp(time, time[::interval], mu)
    var_interp = np.interp(time, time[::interval], var)
    
    return mu_interp, var_interp

def procFlares(files, sector, makefig=True, clobber=False, smoothType='roll_med', progCounter=False):

    FL_id = np.array([])
    FL_t0 = np.array([])
    FL_t1 = np.array([])
    FL_f0 = np.array([])
    FL_f1 = np.array([])
    FL_ed = np.array([])

    for k in range(len(files)):
        print(files[k])
        filename = files[k].split('/')[-1]
        acf_1dt = 0

        with fits.open(files[k], mode='readonly') as hdulist:
            tess_bjd = hdulist[1].data['TIME']
            quality = hdulist[1].data['QUALITY']
            pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
            pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']
            
        # Split data into segments
        dt_limit = 12/24 # 12 hours
        trim = 4/24 # 4 hours
        istart, istop = id_segments(tess_bjd, dt_limit, dt_trim=trim)
        for seg_idx in range(len(istart)):
            tess_bjd_seg = tess_bjd[istart[seg_idx]:istop[seg_idx]]
            quality_seg = quality[istart[seg_idx]:istop[seg_idx]]
            pdcsap_flux_seg = pdcsap_flux[istart[seg_idx]:istop[seg_idx]]
            pdcsap_flux_error_seg = pdcsap_flux_error[istart[seg_idx]:istop[seg_idx]]
            
            ok_cut = quality_seg == 0
            tbl = Table([tess_bjd_seg[ok_cut], pdcsap_flux_seg[ok_cut], pdcsap_flux_error_seg[ok_cut]], 
                         names=('TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'))
            df_tbl = tbl.to_pandas()

            median = np.nanmedian(df_tbl['PDCSAP_FLUX'])

            if smoothType == 'spline':
                smo = IRLSSpline(df_tbl['TIME'].values, df_tbl['PDCSAP_FLUX'].values/median,
                                 df_tbl['PDCSAP_FLUX_ERR'].values/median)
            else:
                acf = xo.autocorr_estimator(tbl['TIME'], tbl['PDCSAP_FLUX']/median,
                                            yerr=tbl['PDCSAP_FLUX_ERR']/median,
                                            min_period=0.1, max_period=27, max_peaks=2)

                if len(acf['peaks']) > 0:
                    acf_1dt = acf['peaks'][0]['period']
                    mask = np.where((acf['autocorr'][0] == acf['peaks'][0]['period']))[0]
                    acf_1pk = acf['autocorr'][1][mask][0]
                    s_window = int(acf_1dt/np.fabs(np.nanmedian(np.diff(df_tbl['TIME']))) / 6)
                else:
                    acf_1dt = (tbl['TIME'][-1] - tbl['TIME'][0])/2
                    s_window = 128

                if smoothType == 'roll_med':
                    smo = df_tbl['PDCSAP_FLUX'].rolling(s_window, center=True).median()
                elif smoothType == 'gauss_proc':
                    smo, var = iterGaussProc(df_tbl['TIME'], df_tbl['PDCSAP_FLUX']/median,
                                            df_tbl['PDCSAP_FLUX_ERR']/median, acf_1dt)

            if makefig:
                fig, axes = plt.subplots(figsize=(8,8))
                axes.errorbar(df_tbl['TIME'], df_tbl['PDCSAP_FLUX']/median,
                              yerr=df_tbl['PDCSAP_FLUX_ERR']/median, 
                              linestyle=None, alpha=0.15, label='PDCSAP_FLUX')
                axes.plot(df_tbl['TIME'], smo/median, label=str(s_window)+'pt median')

                if (acf_1dt > 0):
                    y = np.nanstd(smo/median)*acf_1pk*np.sin(df_tbl['TIME']/acf_1dt*2*np.pi) + 1
                    axes.plot(df_tbl['TIME'], y, lw=2, alpha=0.7,
                             label='ACF='+format(acf_1dt,'6.3f')+'d, pk='+format(acf_1pk,'6.3f'))

            if np.sum(ok_cut) < 1000:
                print('Warning: ' + f + ' contains < 1000 good points')

            sok_cut = np.isfinite(smo)

            FL = FINDflare((df_tbl['PDCSAP_FLUX'][sok_cut] - smo[sok_cut])/median, 
                           df_tbl['PDCSAP_FLUX_ERR'][sok_cut]/median,
                           avg_std=False, N1=4, N2=2, N3=5)

            for j in range(len(FL[0])):
                FL_id = np.append(FL_id, k)
                FL_t0 = np.append(FL_t0, df_tbl['TIME'][sok_cut].values[FL[0][j]])
                FL_t1 = np.append(FL_t1, df_tbl['TIME'][sok_cut].values[FL[1][j]])
                FL_f0 = np.append(FL_f0, median)
                s1, s2 = FL[0][j], FL[1][j]+1
                FL_f1 = np.append(FL_f1, np.nanmax(df_tbl['PDCSAP_FLUX'][sok_cut][s1:s2]))
                ed_val = np.trapz(df_tbl['PDCSAP_FLUX'][sok_cut][s1:s2],
                                  df_tbl['TIME'][sok_cut][s1:s2])
                FL_ed = np.append(FL_ed, ed_val)

                if makefig:
                    axes.scatter(df_tbl['TIME'][sok_cut][s1:s2],
                                 df_tbl['PDCSAP_FLUX'][sok_cut][s1:s2]/median,
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
    flare_out = pd.DataFrame(data={'TIC':ALL_TIC[FL_id],
                                   't0':FL_t0, 't1':FL_t1,
                                   'med':FL_f0, 'peak':FL_f1})
    flare_out.to_csv(sector+ '_flare_out.csv')