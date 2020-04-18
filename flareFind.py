import matplotlib.pylab as plt

import os
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import celerite
from celerite import terms
from scipy.optimize import minimize
from scipy import signal

import flareHelpers as fh

# Debugging stuff, remove later
import traceback

def procFlares(filenames, path, clobberGP=False, makePlots=False):
	if makePlots:
		plots_path = path + 'plots/'
		if not os.path.exists(plots_path):
			os.makedirs(plots_path)

	gp_path = path + 'gp/'

	if not os.path.exists(gp_path):
		os.makedirs(gp_path)

	for k in range(len(filenames)):
		filename = filenames[k]
		file = path + filename

		if makePlots:
			fig, axes = plt.subplots(figsize=(16,16), nrows=4, sharex=True)
		
		print('Processing ' + filename)
		with fits.open(file, mode='readonly') as hdulist:
			try:
				tess_bjd = hdulist[1].data['TIME']
				quality = hdulist[1].data['QUALITY']
				pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
				pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']
			except:
				print('Reading file ' + filename + ' failed')
				continue

		# Cut out poor quality points and convert the remainder to a pandas df
		ok_cut = (quality == 0) & (~np.isnan(tess_bjd)) & (~np.isnan(pdcsap_flux))\
		          & (~np.isnan(pdcsap_flux_error))
		tbl = Table([tess_bjd[ok_cut], pdcsap_flux[ok_cut], pdcsap_flux_error[ok_cut]], 
		             names=('TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'))
		df_tbl = tbl.to_pandas()

		median = np.nanmedian(df_tbl['PDCSAP_FLUX'])

		if makePlots:
			axes[0].plot(tbl['TIME'], tbl['PDCSAP_FLUX'])

		# Estimate the period of the LC with autocorrelation
		acf = fh.autocorr_estimator(tbl['TIME'], tbl['PDCSAP_FLUX']/median, \
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

		# Run GP fit on the lightcurve if we haven't already
		gp_data_file = gp_path + filename + '.gp'
		gp_param_file = gp_path + filename + '.gp.par'
		if os.path.exists(gp_data_file) and not clobberGP:
			# Failed GP regression will produce an empty file
			if os.path.getsize(gp_data_file) == 0:
				print(file + ' failed (previously) during GP regression')
				failed_files.append(filename)
				continue

			print('GP file already exists, loading...')
			time, smo = np.loadtxt(gp_data_file)
		else:
			smo = np.zeros(len(df_tbl['TIME']))
			try:
				if makePlots:
					ax = axes[1]
				else:
					ax = None
				smo, params = iterGP(df_tbl['TIME'], df_tbl['PDCSAP_FLUX']/median, \
				                     df_tbl['PDCSAP_FLUX_ERR']/median, acf_1dt, ax=ax)

				# GP best fit parameters, be sure to write these to the param table
				gp_log_s00 = params[0]
				gp_log_omega00 = params[1]
				gp_log_s01 = params[2]
				gp_log_omega01 = params[3]
				gp_log_q1 = params[4]

				np.savetxt(gp_param_file, params)
				np.savetxt(gp_data_file, (df_tbl['TIME'], smo))

			except:
				traceback.print_exc()
				print(filename + ' failed during GP fitting')
				continue

		# Remove flux jumps by masking out sections of the LC where the GP changes
		# suddenly. Take the rolling STD of the GP, smooth it with a gaussian, and
		# cut out sections where this quantity exceeds 1 sigma. This shouldn't mask
		# out flares because the GP should have smoothed over those

		# 30 minute window (15 data points)
		roll = pd.DataFrame(smo).rolling(15, center=True).std().values.reshape(1, -1)[0]
		mask = np.isfinite(roll)
		roll = roll[mask]
		roll_max = pd.DataFrame(roll).rolling(s_window, center=True).max().values.reshape(1, -1)[0]
		time = df_tbl['TIME'][mask]
		flux = df_tbl['PDCSAP_FLUX'][mask]
		error = df_tbl['PDCSAP_FLUX_ERR'][mask]
		smo = smo[mask]

		roll_cut = 3*np.nanstd(roll)

		mask1 = roll_max < roll_cut

		if makePlots:
			axes[2].scatter(time, roll, s=1)
			axes[2].scatter(time, roll_max, s=1)
			axes[2].axhline(roll_cut, color='r')

		time = time[mask1]
		flux = flux[mask1]
		error = error[mask1]
		smo = smo[mask1]

		# Search for flares in the smoothed lightcurve
		x = np.array(time)
		y = np.array(flux/median - smo)
		yerr =  np.array(error/median)
		FL = fh.FINDflare(y, yerr, avg_std=True, std_window=s_window, N1=3, N2=1, N3=3)

		# Measure properties of detected flares
		axes[3].plot(x, y)
		for j in range(len(FL[0])):
			print('Flare found')
			s1, s2 = FL[0][j], FL[1][j]+1
			axes[3].scatter(time[s1:s2], flux[s1:s2]/median-smo[s1:s2])

		if makePlots:
			axes[0].set_xlabel('Time [BJD - 2457000, days]')
			axes[0].set_ylabel('Flux [e-/s]')
			axes[1].set_xlabel('Time [BJD - 2457000, days]')
			axes[1].set_ylabel('Normalized Flux')
			axes[2].set_xlabel('Time [BJD - 2457000, days]')
			axes[2].set_ylabel('Rolling STD of GP')
			axes[3].set_xlabel('Time [BJD - 2457000, days]')
			axes[3].set_ylabel('Normalized Flux - GP')
			plt.savefig(plots_path + filename + '.png', format='png')

def iterGP(x, y, yerr, period_guess, num_iter=20, ax=None):
    # Here is the kernel we will use for the GP regression
    # It consists of a sum of two stochastically driven damped harmonic
    # oscillators. One of the terms has Q fixed at 1/sqrt(2), which
    # forces it to be non-periodic. There is also a white noise term
    # included.
    
    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(y) / (w0 * Q)
    bounds = dict(log_S0=(-20, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                           bounds=bounds)
    kernel.freeze_parameter('log_Q')

    # A periodic component
    Q = 1.0
    w0 = 2*np.pi/period_guess
    S0 = np.var(y) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                            bounds=bounds)

    # Now calculate the covariance matrix using the initial
    # kernel parameters
    gp = celerite.GP(kernel, mean=np.mean(y))
    gp.compute(x, yerr)

    def neg_log_like(params, y, gp, m):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y[m])

    def grad_neg_log_like(params, y, gp,m ):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y[m])[1]

    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    
    if ax:
    	ax.plot(x, y)

    # Find the best fit kernel parameters. We want to try to ignore the flares
    # when we do the fit. To do this, we will repeatedly find the best fit
    # solution to the kernel model, calculate the covariance matrix, predict
    # the flux and then mask out points based on how far they deviate from
    # the model. After a few passes, this should cause the model to fit mostly
    # to periodic features.
    m = np.ones(len(x), dtype=bool)
    for i in range(num_iter):
        n_pts_prev = np.sum(m)
        gp.compute(x[m], yerr[m])
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method='L-BFGS-B', bounds=bounds, args=(y, gp, m))
        gp.set_parameter_vector(soln.x)
        initial_params = soln.x
        mu = gp.predict(y[m], x, return_cov=False, return_var=False)
        var = np.nanvar(y - mu)
        sig = np.sqrt(var)

        if ax:
        	ax.plot(x, mu)

        m0 = y - mu < sig
        m[m==1] = m0[m==1]
        n_pts = np.sum(m)
        print(n_pts_prev, n_pts)
        if n_pts <= 1000:
            raise ValueError('GP iteration threw out too many points')
            break
        if (n_pts == n_pts_prev):
            break

    gp.compute(x[m], yerr[m])
    mu = gp.predict(y[m], x, return_cov=False, return_var=False)
    
    return mu, gp.get_parameter_vector()