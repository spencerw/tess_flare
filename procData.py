import numpy as np
import pandas as pd
import os
import astropy.units as u

from flareTools import printProgressBar

# Get rotation periods from GP output files
# Crossmatch GAIA and TESS targets
# Calculate luminosities by estimating TESS mag from GAIA color
# Filter flares based on flare model fits

prefix = 'allsky'
path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/' + prefix + '/'
clobber = False

print('Processing data for ' + prefix)

#####################################################################
# Step 1: Grab rotation periods from gp parameter files
#####################################################################
print('Grabbing GP rotation periods')

if not os.path.exists(prefix + '_rot_param_out.csv') or clobber:
    df_param = pd.read_csv(path + 'log/' + prefix + '_param_out.csv')

    log_P_vals = []

    l = len(df_param)
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for idx, row in df_param.iterrows():
        file = row['file']
        try:
            gp_par = np.loadtxt(path + 'gp/' + file + '.gp.par')
            log_P = gp_par[-1] # Natural log!
        except OSError:
            log_P = -15
        log_P_vals.append(log_P)

        printProgressBar(idx, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

    df_param['log_P'] = log_P_vals
    df_param.to_csv(prefix + '_rot_param_out.csv')
    print('    ...Done')
else:
    print('    ...Already done, skipping')
    df_param = pd.read_csv(prefix + '_rot_param_out.csv')

#####################################################################
# Step 2: Crossmatch GAIA and TESS targets
#####################################################################
print('Crossmatching GAIA and TESS targets')

sectors = [pd.read_csv('TESS-Gaia/gaiatess{0}_xmatch_1arsec-result.csv'.format(n+1)) for n in range(27)]
data = pd.concat(sectors, sort=True)

df_param = pd.read_csv(prefix + '_rot_param_out.csv')
unique_tics = np.unique(df_param['TIC'])
data_dd = data.drop_duplicates('ticid')

tess_gaia = data_dd[np.isin(data_dd['ticid'], unique_tics)]
tics = tess_gaia['ticid']
dist = tess_gaia['r_est'].values*u.pc

# Absolute G magnitude (from GAIA dr2 paper)
G_abs_mag = tess_gaia['phot_g_mean_mag'] + 5 + (5*np.log10(tess_gaia['parallax']/1000))

iso_table1 = pd.read_csv('output221284170243.dat.txt', comment='#', delim_whitespace=True)
iso_table2 = pd.read_csv('output768246532491.dat.txt', comment='#', delim_whitespace=True)
iso_table1['Gmag'] = iso_table2['Gmag']
iso_table1['G_BPbrmag'] = iso_table2['G_BPbrmag']
iso_table1['G_BPftmag'] = iso_table2['G_BPftmag']
iso_table1['G_RPmag'] = iso_table2['G_RPmag']
iso = iso_table1

T = iso['TESSmag'].values[::-1]
G_B = iso['G_BPbrmag'].values[::-1]
G_R = iso['G_RPmag'].values[::-1]
G_mag = iso['Gmag'].values[::-1]
color = G_mag - G_R

# Interpolate gaia color onto isochrone to get TESS mag
G_min_Grp = tess_gaia['phot_g_mean_mag'] - tess_gaia['phot_rp_mean_mag']

# The cutting gets a little funky here. Need to cleanly cut out the main sequence part of the isochrone, otherwise
# the interpolation doesn't work correctly. Also need to grab the next point past the end of the color cut, otherwise
# the interpolated model ends up with a flat spot in the CMD on the right side of the main sequence
color_min, color_max = 0.2, 1.4
iso_width = 0.4 # How far above and below the isochrone to go to select MS stars
mask_iso = (G_mag > 2) & (G_mag - G_R > color_min) & (G_mag - G_R < 1.45) # Go a little past the color limit to get the next point in the isochrone model
mask_real = (G_min_Grp > color_min) & (G_min_Grp < color_max)
TESS_mag_int = np.interp(G_min_Grp, G_mag[mask_iso] - G_R[mask_iso], T[mask_iso])
G_abs_mag_int = np.interp(G_min_Grp[mask_real], G_mag[mask_iso] - G_R[mask_iso], G_mag[mask_iso])

# Convert TESS mag to luminosity
Tf0 = 4.03e-6*u.erg/u.s/u.cm**2 # Zero point TESS flux (from Sullivan 2017)

# TESS apparent magnitude
m_t = TESS_mag_int + 5*np.log10(dist.value) - 5
f = 10**(-m_t/2.5)*Tf0
L = 4*np.pi*(dist.to(u.cm))**2*f

ms_mask = (G_min_Grp[mask_real] > color_min) & (G_min_Grp[mask_real] < color_max) & (G_abs_mag[mask_real] < G_abs_mag_int + iso_width) & (G_abs_mag[mask_real] > G_abs_mag_int - iso_width)

tess_gaia_table = pd.DataFrame({'TIC':tics, 'r_est':dist, 'lum':L, 'G_BPbrmag':tess_gaia['phot_bp_mean_mag'], \
                      'G_RPmag':tess_gaia['phot_rp_mean_mag'], 'G_mag':tess_gaia['phot_g_mean_mag'], 'G_abs_mag':G_abs_mag,\
                       'source_id':tess_gaia['source_id'], 'ra':tess_gaia['ra'], 'dec':tess_gaia['dec'], 'is_ms': np.isin(tics, tics[mask_real][ms_mask])})

print(str(len(tess_gaia_table)) + ' targets found in TESS-Gaia crossmatch')

#####################################################################
# Step 3: Merge TESS-Gaia table with flare table
#####################################################################
print('Filtering flare targets with TESS-Gaia crossmatch')

df_flare = pd.read_csv(path + 'log/' + prefix + '_flare_out.csv')
df_flare_g = pd.merge(tess_gaia_table, df_flare, on='TIC', how='inner')
df_param_g = pd.merge(tess_gaia_table, df_param, on='TIC', how='inner')

# Throw out bad luminosity measurements
df_flare_g = df_flare_g.dropna(subset=['lum'])
ed = (df_flare_g['ed'].values*u.day).to(u.s)
energy = (ed*df_flare_g['lum']*u.erg/u.s).value
df_flare_g['energy'] = energy

ed_err = (df_flare_g['ed_err'].values*u.day).to(u.s)
energy_err = (ed_err*df_flare_g['lum']*u.erg/u.s).value
df_flare_g['energy_err'] = energy_err

# TODO: Filter out MS stars by color and mag

print(str(len(df_flare_g)) + ' flare events and ' + str(len(df_param_g)) + ' light curves in TESS-Gaia crossmatch')

#####################################################################
# Step 4: Cut out bad flare detections
#####################################################################
print('Filtering out bad flare candidates')

df = df_flare_g
mask = (df['skew'] > 0.5) & \
       (df['f_chisq'] > 0) & \
       (df['f_fwhm_win'] < 0.1) &(df['g_fwhm_win'] < 0.1) & \
       (df['ed'] > 0) & (df['tpeak'] > df['t0']) & (df['tpeak'] < df['t1']) & \
       (df['cover'] > 0.9) & \
       (df['f_chisq'] < 0.7*df['g_chisq'])

df = df[mask]
print(str(len(df)) + ' flare events from ' + str(len(np.unique(df['TIC']))) + ' targets in ' + str(len(np.unique(df['file']))) + ' lightcurves')

df.to_csv(prefix + '_flare_out.csv')
df_param_g.to_csv(prefix + '_param_out.csv')
