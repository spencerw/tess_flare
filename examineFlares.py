import matplotlib.pylab as plt

import os
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import astropy.units as u

import flareHelpers as fh

prefix = 'test'
path = 'test_files/'
plots_path = path + 'plots_filt/'

if not os.path.exists(plots_path):
	os.makedirs(plots_path)
else:
	files = glob.glob(plots_path + '*')
	for f in files:
		os.remove(f)

df = pd.read_csv(path + prefix + '_flare_out.csv')
df_param = pd.read_csv(path + prefix + '_param_out.csv')
mask = (df['skew'] > 0.5) & (df['f_chisq'] < 2) & (df['f_chisq'] > 0) & (df['f_fwhm_win'] < 0.1) & \
       (df['tpeak'] > df['t0']) & (df['tpeak'] < df['t1']) & (df['cover'] > 0.9) & (df['f_chisq']/df['g_chisq'] < 0.7)
print(len(df), len(df[mask]))

df = df[mask]
flare_files = np.unique(df['file'])

for file in flare_files:
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