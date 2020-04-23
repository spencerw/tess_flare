import matplotlib.pylab as plt

import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u

import flareHelpers as fh

prefix = '1to13'
path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/sec1to13/'
plots_path = path + 'plots_filt/'
log_path = path + 'log/'

if not os.path.exists(plots_path):
	os.makedirs(plots_path)

df = pd.read_csv(log_path + prefix + '_flare_out.csv')
df_param = pd.read_csv(log_path + prefix + '_param_out.csv')
df = df[df['f_chisq'] > 0]
flare_files = np.unique(df['file'])

print(len(df), len(flare_files))

for file in flare_files:
	if os.path.exists(plots_path + file + '.png'):
	    continue
	flares = df[df['file'] == file]
	par = df_param[df_param['file'] == file].iloc[0]
	
	median = par['med']
	time_smo, smo = np.loadtxt(path + 'gp/' + file + '.gp')

	# First plot the LC with the CPA points overlaid

	with fits.open(path + file, mode='readonly') as hdulist:
		tess_bjd = hdulist[1].data['TIME']
		quality = hdulist[1].data['QUALITY']
		pdcsap_flux = hdulist[1].data['PDCSAP_FLUX']
		pdcsap_flux_error = hdulist[1].data['PDCSAP_FLUX_ERR']

	fig, axes = plt.subplots(figsize=(16,4))
	axes.plot(tess_bjd, pdcsap_flux, zorder=1)

	ok_cut = (quality == 0) & (~np.isnan(tess_bjd)) & (~np.isnan(pdcsap_flux)) \
		          & (~np.isnan(pdcsap_flux_error))

	time = tess_bjd[ok_cut]
	flux = pdcsap_flux[ok_cut]
	error = pdcsap_flux_error[ok_cut]

	for idx in range(len(flares)):
		fl = flares.iloc[idx]
		t0, t1 = fl['t0'], fl['t1']
		mask = (time >= t0) & (time < t1)
		axes.scatter(time[mask], flux[mask], zorder=2)

	axes.set_xlabel('Time [BJD - 2457000, days]')
	axes.set_ylabel('Flux [e-/s]')
	fig.savefig(plots_path + file + '.png', format='png')

	# Now plot individual flares

	fig, axes = plt.subplots(figsize=(16,16), nrows=4, ncols=4)

	for idx in range(len(flares)):
		fl = flares.iloc[idx]
		if idx > 15:
			break
		row_idx = idx//4
		col_idx = idx%4

		t0, t1 = fl['t0'], fl['t1']
		mask = (time >= t0) & (time < t1)
		axes[row_idx][col_idx].errorbar(time[mask], flux[mask]/median - smo[mask], error[mask]/median)

		xmodel = np.linspace(t0, t1)
		ymodel = fh.aflare1(xmodel, fl['tpeak'], fl['fwhm'], fl['f_amp'])
		axes[row_idx][col_idx].plot(xmodel, ymodel, label=r'$\chi_{f}$ = ' + '{:.3f}'.format(fl['f_chisq']) \
			                           + '\n FWHM/window = ' + '{:.2f}'.format(fl['f_fwhm_win']))
		ymodel = fh.gaussian(xmodel, fl['mu'], fl['std'], fl['g_amp'])
		axes[row_idx][col_idx].plot(xmodel, ymodel, label=r'$\chi_{g}$ = ' + '{:.3f}'.format(fl['g_chisq']) \
			                           + '\n FWHM/window = ' + '{:.2f}'.format(fl['g_fwhm_win']))
		axes[row_idx][col_idx].legend()
		axes[row_idx][col_idx].set_title('Skew = ' + '{:.3f}'.format(fl['skew']))

	fig.savefig(plots_path + file + '_flares.png', format='png')
	plt.close('all')
