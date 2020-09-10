import numpy as np
import pandas as pd
import os
import astropy.units as u

from flareTools import printProgressBar

# Get rotation periods from GP output files
# Crossmatch GAIA and TESS targets
# Calculate luminosities by estimating TESS mag from GAIA color
# Filter flares based on flare model fits

prefix = '14to26'
path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/sec1to13/'
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

sectors = [pd.read_csv('TESS-Gaia/gaiatess{0}_xmatch_1arsec-result.csv'.format(n+1)) for n in range(26)]
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
GminR = G_B - G_R

# Do a cut to get rid of giants
mask = (G_B > 4) & (GminR > 0) & (GminR < 4.5)

# Interpolate gaia color onto isochrone to get TESS mag
Gbp_min_Grp = tess_gaia['phot_g_mean_mag'] - tess_gaia['phot_rp_mean_mag']
TESS_mag_int = np.interp(Gbp_min_Grp, G_B[mask] - G_R[mask], T[mask])

# Convert TESS mag to luminosity
Tf0 = 4.03e-6*u.erg/u.s/u.cm**2 # Zero point TESS flux (from Sullivan 2017)

# TESS apparent magnitude
m_t = TESS_mag_int + 5*np.log10(dist.value) - 5
f = 10**(-m_t/2.5)*Tf0
L = 4*np.pi*(dist.to(u.cm))**2*f

tess_gaia_table = pd.DataFrame({'TIC':tics, 'r_est':dist, 'lum':L, 'G_BPbrmag':tess_gaia['phot_bp_mean_mag'], \
                      'G_RPmag':tess_gaia['phot_rp_mean_mag'], 'G_mag':tess_gaia['phot_g_mean_mag'], 'G_abs_mag':G_abs_mag,\
                       'source_id':tess_gaia['source_id'], 'ra':tess_gaia['ra'], 'dec':tess_gaia['dec']})

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
df_flare_g = df_flare_g.dropna(subset=['lum'])
ed = (df_flare_g['ed'].values*u.day).to(u.s)
energy = (ed*df_flare_g['lum']*u.erg/u.s).value
df_flare_g['energy'] = energy

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
       (df['f_chisq'] < df['g_chisq'])

df = df[mask]
print(str(len(df)) + ' flare events from ' + str(len(np.unique(df['TIC']))) + ' targets in ' + str(len(np.unique(df['file']))) + ' lightcurves')

df.to_csv(prefix + '_flare_out.csv')
df_param_g.to_csv(prefix + '_param_out.csv')