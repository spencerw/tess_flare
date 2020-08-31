#import matplotlib.pylab as plt

import os
import autograd.numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
import celerite
from celerite import terms
from scipy.optimize import minimize, curve_fit
import astropy.units as u
import time as timing

import flareHelpers as fh

# Debugging stuff, remove later
import traceback

def procFlares(prefix, filenames, path, clobberGP=False, makePlots=False, writeLog=True):
	if makePlots:
		plots_path = path + 'plots/'
		if not os.path.exists(plots_path):
			os.makedirs(plots_path)

	gp_path = path + 'gp/'

	#if not os.path.exists(gp_path):
		#os.makedirs(gp_path)

	log_path = path + 'log/'

	#if not os.path.exists(log_path):
		#os.makedirs(log_path)

	if writeLog:
		if os.path.exists(log_path + prefix + '.log'):
			os.remove(log_path + prefix + '.log')

	# Columns for flare table
	FL_files = np.array([])
	FL_TICs = np.array([])
	FL_id = np.array([])
	FL_t0 = np.array([])
	FL_t1 = np.array([])
	FL_f0 = np.array([])
	FL_f1 = np.array([])
	FL_ed = np.array([])
	FL_ed_err = np.array([])
	FL_skew = np.array([])
	FL_cover = np.array([])
	FL_mu = np.array([])
	FL_std = np.array([])
	FL_g_amp = np.array([])
	FL_mu_err = np.array([])
	FL_std_err = np.array([])
	FL_g_amp_err = np.array([])
	FL_tpeak = np.array([])
	FL_fwhm = np.array([])
	FL_f_amp = np.array([])
	FL_tpeak_err = np.array([])
	FL_fwhm_err = np.array([])
	FL_f_amp_err = np.array([])
	FL_g_chisq = np.array([])
	FL_f_chisq = np.array([])
	FL_g_fwhm_win = np.array([])
	FL_f_fwhm_win = np.array([])

	# Columns for param table
	P_median = np.array([])
	P_s_window = np.array([])
	P_acf_1dt = np.array([])

	failed_files = []

	for k in range(len(filenames)):
		start_time = timing.time()
		filename = filenames[k]
		TIC = int(filename.split('-')[-3])
		file = path + filename

		if makePlots:
			fig, axes = plt.subplots(figsize=(16,16), nrows=4, sharex=True)
		
		print('Processing ' + filename)
		gp_data_file = gp_path + filename + '.gp'
		gp_param_file = gp_path + filename + '.gp.par'
		median = -1
		s_window = -1
		acf_1dt = -1
		with fits.open(file, mode='readonly') as hdulist:
			try:
				tess_bjd = hdulist[1].data['TIME']
				quality = hdulist[1].data['QUALITY']
				pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
				pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']
			except:
				P_median = np.append(P_median, median)
				P_s_window = np.append(P_s_window, s_window)
				P_acf_1dt = np.append(P_acf_1dt, acf_1dt)
				failed_files.append(filename)
				np.savetxt(gp_data_file, ([]))
				print('Reading file ' + filename + ' failed')
				continue

		if makePlots:
			axes[0].plot(tess_bjd, pdcsap_flux)

		# Cut out poor quality points
		ok_cut = (quality == 0) & (~np.isnan(tess_bjd)) & (~np.isnan(pdcsap_flux))\
		          & (~np.isnan(pdcsap_flux_error))

		tbl = Table([tess_bjd[ok_cut], pdcsap_flux[ok_cut], \
			         pdcsap_flux_error[ok_cut]], 
		             names=('TIME', 'PDCSAP_FLUX', 'PDCSAP_FLUX_ERR'))
		df_tbl = tbl.to_pandas()

		median = np.nanmedian(df_tbl['PDCSAP_FLUX'])

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

		P_median = np.append(P_median, median)
		P_s_window = np.append(P_s_window, s_window)
		P_acf_1dt = np.append(P_acf_1dt, acf_1dt)

		# Run GP fit on the lightcurve if we haven't already
		if os.path.exists(gp_data_file) and not clobberGP:
			# Failed GP regression will produce an empty file
			if os.path.getsize(gp_data_file) == 0:
				print(file + ' failed (previously) during GP regression')
				failed_files.append(filename)
				continue

			print('GP file already exists, loading...')
			times, smo, var = np.loadtxt(gp_data_file)
		else:
			smo = np.zeros(len(df_tbl['TIME']))
			try:
				if makePlots:
					ax = axes[1]
				else:
					ax = None
				times, smo, var, params = iterGP_rotation(df_tbl['TIME'].values, df_tbl['PDCSAP_FLUX'].values/median, \
				                          df_tbl['PDCSAP_FLUX_ERR'].values/median, acf_1dt, acf_1pk, ax=ax)


				#np.savetxt(gp_param_file, params['logs2'], params['logamp'], params['logperiod'], \
                                #           params['logq0'], params['logdeltaq'], params['mix'], params['period'])
				np.savetxt(gp_param_file, params)
				np.savetxt(gp_data_file, (times, smo, var))

			except:
				traceback.print_exc()
				failed_files.append(filename)
				np.savetxt(gp_data_file, ([]))
				print(filename + ' failed during GP fitting')
				continue

		# The GP is produced from a downsampled lightcurve. Need to interpolate to
		# compare GP and full LC

		smo_int = np.interp(tbl['TIME'], times, smo)

		# Search for flares in the smoothed lightcurve
		x = np.array(tbl['TIME'])
		y = np.array(tbl['PDCSAP_FLUX']/median - smo_int)
		yerr =  np.array(tbl['PDCSAP_FLUX_ERR']/median)

		FL = fh.FINDflare(y, yerr, avg_std=True, std_window=s_window, N1=3, N2=1, N3=3)
		
		if makePlots:
			axes[3].plot(x, y, zorder=1)
			for j in range(len(FL[0])):
				s1, s2 = FL[0][j], FL[1][j]+1
				axes[3].scatter(x[s1:s2], y[s1:s2], zorder=2)

		# Measure properties of detected flares
		if makePlots:
			fig_fl, axes_fl = plt.subplots(figsize=(16,16), nrows=4, ncols=4)

		for j in range(len(FL[0])):
			s1, s2 = FL[0][j], FL[1][j]+1
			tstart, tstop = x[s1], x[s2]
			dx_fac  = 10
			dx = tstop - tstart
			x1 = tstart - dx*dx_fac/2
			x2 = tstop + dx*dx_fac/2
			mask = (x > x1) & (x < x2)
			
			# Mask out other flare detections when fitting models
			other_mask = np.ones(len(x), dtype=bool)
			for i in range(len(FL[0])):
			    s1other, s2other = FL[0][i], FL[1][i]+1
			    if i == j:
			        continue
			    other_mask[s1other:s2other] = 0

			popt1, pstd1, g_chisq, popt2, pstd2, f_chisq, skew, cover = \
			    fitFlare(x[other_mask], y[other_mask], yerr[other_mask], x1, x2)

			mu, std, g_amp = popt1[0], popt1[1], popt1[2]
			mu_err, std_err, g_amp_err = pstd1[0], pstd1[1], pstd1[2]
			    
			tpeak, fwhm, f_amp = popt2[0], popt2[1], popt2[2]
			tpeak_err, fwhm_err, f_amp_err = pstd2[0], pstd2[1], pstd2[2]

			f_fwhm_win = fwhm/(x2 - x1)
			g_fwhm_win = std/(x2 - x1)

			ed, ed_err = measureED(x, y, yerr, tpeak, fwhm)

			FL_files = np.append(FL_files, filename)
			FL_TICs = np.append(FL_TICs, TIC)
			FL_t0 = np.append(FL_t0, x1)
			FL_t1 = np.append(FL_t1, x2)
			FL_f0 = np.append(FL_f0, np.nanmedian(tbl['PDCSAP_FLUX'][s1:s2]))
			FL_f1 = np.append(FL_f1, np.nanmax(tbl['PDCSAP_FLUX'][s1:s2]))
			FL_ed = np.append(FL_ed, ed)
			FL_ed_err = np.append(FL_ed_err, ed_err)

			FL_skew = np.append(FL_skew, skew)
			FL_cover = np.append(FL_cover, cover)
			FL_mu = np.append(FL_mu, mu)
			FL_std = np.append(FL_std, std)
			FL_g_amp = np.append(FL_g_amp, g_amp)
			FL_mu_err = np.append(FL_mu_err, mu_err)
			FL_std_err = np.append(FL_std_err, std_err)
			FL_g_amp_err = np.append(FL_g_amp_err, g_amp_err)

			FL_tpeak = np.append(FL_tpeak, tpeak)
			FL_fwhm = np.append(FL_fwhm, fwhm)
			FL_f_amp = np.append(FL_f_amp, f_amp)
			FL_tpeak_err = np.append(FL_tpeak_err, tpeak_err)
			FL_fwhm_err = np.append(FL_fwhm_err, fwhm_err)
			FL_f_amp_err = np.append(FL_f_amp_err, f_amp_err)

			FL_g_chisq = np.append(FL_g_chisq, g_chisq)
			FL_f_chisq = np.append(FL_f_chisq, f_chisq)

			FL_g_fwhm_win = np.append(FL_g_fwhm_win, g_fwhm_win)
			FL_f_fwhm_win = np.append(FL_f_fwhm_win, f_fwhm_win)

			if makePlots and j < 15:
				row_idx = j//4
				col_idx = j%4
				axes_fl[row_idx][col_idx].errorbar(x[mask], y[mask], yerr=yerr[mask])
				axes_fl[row_idx][col_idx].scatter(x[s1:s2], y[s1:s2])

				if popt1[0] > 0:
					xmodel = np.linspace(x1, x2)
					ymodel = fh.aflare1(xmodel, tpeak, fwhm, f_amp)
					axes_fl[row_idx][col_idx].plot(xmodel, ymodel, label=r'$\chi_{f}$ = ' + '{:.3f}'.format(f_chisq) \
						                           + '\n FWHM/window = ' + '{:.2f}'.format(f_fwhm_win))
					ymodel = fh.gaussian(xmodel, mu, std, g_amp)
					axes_fl[row_idx][col_idx].plot(xmodel, ymodel, label=r'$\chi_{g}$ = ' + '{:.3f}'.format(g_chisq) \
						                           + '\n FWHM/window = ' + '{:.2f}'.format(g_fwhm_win))
					axes_fl[row_idx][col_idx].axvline(tpeak - fwhm/2, linestyle='--')
					axes_fl[row_idx][col_idx].axvline(tpeak + fwhm/2, linestyle='--')
					axes_fl[row_idx][col_idx].legend()
					axes_fl[row_idx][col_idx].set_title('Skew = ' + '{:.3f}'.format(skew))

		if makePlots:
			fig.suptitle(filename)
			axes[0].set_xlabel('Time [BJD - 2457000, days]')
			axes[0].set_ylabel('Flux [e-/s]')
			axes[1].set_xlabel('Time [BJD - 2457000, days]')
			axes[1].set_ylabel('Normalized Flux')
			axes[2].set_xlabel('Time [BJD - 2457000, days]')
			axes[2].set_ylabel('Rolling STD of GP')
			axes[3].set_xlabel('Time [BJD - 2457000, days]')
			axes[3].set_ylabel('Normalized Flux - GP')
			fig.savefig(plots_path + filename + '.png', format='png')

			if len(FL[0] > 0):
				fig_fl.suptitle(filename)
				fig_fl.savefig(plots_path + filename + '_flares.png', format='png')
             
			plt.clf()

		if writeLog:
			with open(log_path + prefix + '.log', 'a') as f:
				time_elapsed = timing.time() - start_time
				num_flares = len(FL[0])

				f.write('{:^15}'.format(str(k+1) + '/' + str(len(filenames))) + \
				        '{:<60}'.format(filename) + '{:<20}'.format(time_elapsed) + \
				        '{:<10}'.format(num_flares) + '\n')

		# Periodically write to the flare table file and param table file
		l = k+1
		ALL_TIC = pd.Series(filenames).str.split('-', expand=True).iloc[:,-3].astype('int')
		ALL_FILES = pd.Series(filenames).str.split('/', expand=True).iloc[:,-1]

		flare_out = pd.DataFrame(data={'file':FL_files,'TIC':FL_TICs, 't0':FL_t0, 't1':FL_t1, \
			                           'med_flux':FL_f0, 'peak_flux':FL_f1, 'ed':FL_ed, \
			                           'ed_err':FL_ed_err, 'skew':FL_skew, 'cover':FL_cover, \
			                           'mu':FL_mu, 'std':FL_std, 'g_amp': FL_g_amp, 'mu_err':FL_mu_err, \
			                           'std_err':FL_std_err, 'g_amp_err':FL_g_amp_err,'tpeak':FL_tpeak, \
			                           'fwhm':FL_fwhm, 'f_amp':FL_f_amp, 'tpeak_err':FL_tpeak_err, \
			                           'fwhm_err':FL_fwhm_err, 'f_amp_err':FL_f_amp_err,'f_chisq':FL_f_chisq, \
			                           'g_chisq':FL_g_chisq, 'f_fwhm_win':FL_f_fwhm_win, 'g_fwhm_win':FL_g_fwhm_win})
		flare_out.to_csv(log_path + prefix + '_flare_out.csv', index=False)

		param_out = pd.DataFrame(data={'file':ALL_FILES[:l], 'TIC':ALL_TIC[:l], 'med':P_median[:l], \
			                           's_window':P_s_window[:l], 'acf_1dt':P_acf_1dt[:l]})
		param_out.to_csv(log_path + prefix + '_param_out.csv', index=False)

	for k in range(len(failed_files)):
		print(failed_files[k])

def fitFlare(x, y, yerr, tstart, tstop, skew_fac=10):
	mask = (x > tstart) & (x < tstop)
	mu0 = (tstart + tstop)/2
	sig0 = (tstop - tstart)/2
	A0 = np.max(y)*100
	skew = 0

	try:
		# Fit a gaussian to the segment
		popt1, pcov1 = curve_fit(fh.gaussian, x[mask], y[mask], p0=(mu0, sig0, A0), sigma=yerr[mask])
		y_model = fh.gaussian(x[mask], popt1[0], popt1[1], popt1[2])
		chi1 = fh.redChiSq(y_model, y[mask], yerr[mask], len(y[mask]) - 3)

		# Fit the Davenport 2014 flare model to the segment
		popt2, pcov2 = curve_fit(fh.aflare1, x[mask], y[mask], p0=(mu0, sig0, A0), sigma=yerr[mask])
		y_model = fh.aflare1(x[mask], popt2[0], popt2[1], popt2[2])
		chi2 = fh.redChiSq(y_model, y[mask], yerr[mask], len(y[mask]) - 3)

		# If the flare model fit worked, calculate the skew by centering on the peak of the aflare model
		# Use a window scaled to the FWHM of the flare model for integration
		mu = popt2[0] #np.trapz(x[mask]*A*y[mask], x[mask])
		f_hwhm = popt2[1]/2
		t1_skew, t2_skew = mu - skew_fac*f_hwhm, mu + skew_fac*f_hwhm
		skew_mask = (x > t1_skew) & (x < t2_skew)

		# Measure the skew by treating time = x and flux = p(x). Calculate the
		# third moment of p(x)
		A = 1/np.trapz(y[skew_mask], x[skew_mask])
		var = np.trapz((x[skew_mask] - mu)**2*A*y[skew_mask], x[skew_mask])
		stddev = np.sqrt(np.fabs(var))
		skew = np.trapz((x[skew_mask] - mu)**3*A*y[skew_mask], x[skew_mask])/stddev**3
	except:
		traceback.print_exc()
		empty = np.zeros(3)
		return empty, empty, -1, empty, empty, -1, 0, 0

	n_pts = len(x[mask])
	n_pts_true = np.floor(((tstop-tstart)*u.d).to(u.min).value/2)
	coverage = n_pts/n_pts_true

	return popt1, np.sqrt(pcov1.diagonal()), chi1, popt2, np.sqrt(pcov2.diagonal()), chi2, skew, coverage

def measureED(x, y, yerr, tpeak, fwhm, num_fwhm=10):
    '''
    Measure the equivalent duration of a flare in a smoothed light
    curve. FINDflare typically doesnt identify the entire flare, so
    integrate num_fwhm/2*fwhm away from the peak. As long as the light 
    curve is flat away from the flare, the region around the flare should
    not significantly contribute.

    Parameters
    ----------
    x : numpy array
        time values from the entire light curve
    y : numpy array
        flux values from the entire light curve
    yerr : numpy array
        error in the flux values
    tpeak : float
        Peak time of the flare detection
    fwhm : float
        Full-width half maximum of the flare
    num_fwhm : float, optional
        Size of the integration window in units of fwhm
    Returns
    -------
        ED - Equivalent duration of the flare
        ED_err - The uncertainty in the equivalent duration
    '''

    try:
        width = fwhm*num_fwhm
        istart = np.argwhere(x > tpeak - width/2)[0]
        ipeak = np.argwhere(x > tpeak)[0]
        istop = np.argwhere(x > tpeak + width/2)[0]
    
        dx = np.diff(x)
        x = x[:-1]
        y = y[:-1]
        yerr = yerr[:-1]
        mask = (x > x[istart]) & (x < x[istop])
        ED = np.trapz(y[mask], x[mask])
        ED_err = np.sqrt(np.sum((dx[mask]*yerr[mask])**2))

    except IndexError:
        return -1, -1
    
    return ED, ED_err

def iterGP(x, y, yerr, period_guess, acf_1pk, num_iter=20, ax=None, n_samp=4000):
    # Here is the kernel we will use for the GP regression
    # It consists of a sum of two stochastically driven damped harmonic
    # oscillators. One of the terms has Q fixed at 1/sqrt(2), which
    # forces it to be non-periodic. There is also a white noise term
    # included.

    # Do some aggressive sigma clipping
    m = np.ones(len(x), dtype=bool)
    while True:
        mu = np.mean(y[m])
        sig = np.std(y[m])
        m0 = y - mu < 3 * sig
        if np.all(m0 == m):
            break
        m = m0
    x_clip, y_clip, yerr_clip = x[m], y[m], yerr[m]

    if len(x_clip) < n_samp:
        n_samp = len(x_clip)

    # Randomly select n points from the light curve for the GP fit
    x_ind_rand = np.random.choice(len(x_clip), n_samp, replace=False)
    x_ind = x_ind_rand[np.argsort(x_clip[x_ind_rand])]

    x_gp = x_clip[x_ind]
    y_gp = y_clip[x_ind]
    yerr_gp = yerr_clip[x_ind]
    
    # A non-periodic component
    Q = 1.0 / np.sqrt(2.0)
    w0 = 3.0
    S0 = np.var(y_gp) / (w0 * Q)
    bounds = dict(log_S0=(-20, 15), log_Q=(-15, 15), log_omega0=(-15, 15))
    kernel = terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                           bounds=bounds)
    kernel.freeze_parameter('log_Q')

    # A periodic component
    Q = 1.0
    w0 = 2*np.pi/period_guess
    S0 = np.var(y_gp) / (w0 * Q)
    kernel += terms.SHOTerm(log_S0=np.log(S0), log_Q=np.log(Q), log_omega0=np.log(w0),
                            bounds=bounds)

    # Now calculate the covariance matrix using the initial
    # kernel parameters
    gp = celerite.GP(kernel, mean=np.mean(y_gp))
    gp.compute(x_gp, yerr_gp)

    def neg_log_like(params, y, gp, m):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y[m])

    def grad_neg_log_like(params, y, gp,m ):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y[m])[1]

    bounds = gp.get_parameter_bounds()
    initial_params = gp.get_parameter_vector()
    
    if ax:
    	ax.plot(x_gp, y_gp)

    # Find the best fit kernel parameters. We want to try to ignore the flares
    # when we do the fit. To do this, we will repeatedly find the best fit
    # solution to the kernel model, calculate the covariance matrix, predict
    # the flux and then mask out points based on how far they deviate from
    # the model. After a few passes, this should cause the model to fit mostly
    # to periodic features.
    m = np.ones(len(x_gp), dtype=bool)
    for i in range(num_iter):
        n_pts_prev = np.sum(m)
        gp.compute(x_gp[m], yerr_gp[m])
        soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                        method='L-BFGS-B', bounds=bounds, args=(y_gp, gp, m))
        gp.set_parameter_vector(soln.x)
        initial_params = soln.x
        mu, var = gp.predict(y_gp[m], x_gp, return_var=True)
        sig = np.sqrt(var + yerr_gp**2)

        if ax:
        	ax.plot(x_gp, mu)

        m0 = y_gp - mu < sig
        m[m==1] = m0[m==1]
        n_pts = np.sum(m)
        print(n_pts, n_pts_prev)
        if n_pts <= 10:
            raise ValueError('GP iteration threw out too many points')
            break
        if (n_pts == n_pts_prev):
            break

    gp.compute(x_gp[m], yerr_gp[m])
    mu, var = gp.predict(y_gp[m], x_gp, return_var=True)
    
    return x_gp, mu, var, gp.get_parameter_vector()

# Same as iterGP, but uses the RotationTerm kernel from the exoplanet package
def iterGP_rotation(x, y, yerr, period_guess, acf_1pk, num_iter=20, ax=None, n_samp=4000):
	# Here is the kernel we will use for the GP regression
	# It consists of a sum of two stochastically driven damped harmonic
	# oscillators. One of the terms has Q fixed at 1/sqrt(2), which
	# forces it to be non-periodic. There is also a white noise term
	# included.
	# Do some aggressive sigma clipping
	m = np.ones(len(x), dtype=bool)
	while True:
		mu = np.mean(y[m])
		sig = np.std(y[m])
		m0 = y - mu < 3 * sig
		if np.all(m0 == m):
			break
		m = m0
	x_clip, y_clip, yerr_clip = x[m], y[m], yerr[m]

	if len(x_clip) < n_samp:
		n_samp = len(x_clip)

	# Randomly select n points from the light curve for the GP fit
	x_ind_rand = np.random.choice(len(x_clip), n_samp, replace=False)
	x_ind = x_ind_rand[np.argsort(x_clip[x_ind_rand])]

	x_gp = x_clip[x_ind]
	y_gp = y_clip[x_ind]
	yerr_gp = yerr_clip[x_ind]

	gp = fh.get_rotation_gp(x_gp, y_gp, yerr_gp, period_guess, 0, 13)
	# Now calculate the covariance matrix using the initial
	# kernel parameters
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
		ax.plot(x_gp, y_gp)

	# Find the best fit kernel parameters. We want to try to ignore the flares
	# when we do the fit. To do this, we will repeatedly find the best fit
	# solution to the kernel model, calculate the covariance matrix, predict
	# the flux and then mask out points based on how far they deviate from
	# the model. After a few passes, this should cause the model to fit mostly
	# to periodic features.
	m = np.ones(len(x_gp), dtype=bool)
	for i in range(num_iter):
		n_pts_prev = np.sum(m)
		gp.compute(x_gp[m], yerr_gp[m])
		soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
		           method='L-BFGS-B', bounds=bounds, args=(y_gp, gp, m))
		gp.set_parameter_vector(soln.x)
		initial_params = soln.x
		mu, var = gp.predict(y_gp[m], x_gp, return_var=True)
		sig = np.sqrt(var + yerr_gp**2)

		m0 = y_gp - mu < sig
		m[m==1] = m0[m==1]
		n_pts = np.sum(m)
		if n_pts <= 10:
			raise ValueError('GP iteration threw out too many points')
			break
		if (n_pts_prev - n_pts) <= 3:
			break

	gp.compute(x_gp[m], yerr_gp[m])
	mu, var = gp.predict(y_gp[m], x_gp, return_var=True)

	return x_gp, mu, var, gp.get_parameter_vector()

def predict(x, y, yerr, m, period_guess):
	with pm.Model() as model:
		# The parameters of the RotationTerm kernel
		logamp = pm.Normal("logamp", mu=np.log(np.var(y[m])), sd=5.0)
		logperiod = pm.Normal("logperiod", mu=np.log(period_guess), sd=5.0)
		logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
		logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
		mix = pm.Uniform("mix", lower=0.0, upper=1.0)

		 # Track the period as a deterministic
		period = pm.Deterministic("period", tt.exp(logperiod))

		# Set up the Gaussian Process model
		kernel = xo.gp.terms.RotationTerm(
		    log_amp=logamp,
		    period=period,
		    log_Q0=logQ0,
		    log_deltaQ=logdeltaQ,
		    mix=mix
		)
		gp = xo.gp.GP(kernel, x[m], yerr[m]**2, J=4)

		# Compute the Gaussian Process likelihood and add it into the
		# the PyMC3 model as a "potential"
		pm.Potential("loglike", gp.log_likelihood(y[m] - 1))

		# Optimize to find the maximum a posteriori parameters
		map_soln = xo.optimize(start=model.test_point, progress_bar=False)

		mu, var = xo.utils.eval_in_model(gp.predict(x, return_var=True), map_soln)

	return mu, var, map_soln
