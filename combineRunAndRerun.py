import pandas as pd
import numpy as np

df = pd.read_csv('7.gauss_flare_out.csv')
df_rerun = pd.read_csv('7.rerun.gauss_flare_out.csv')

for TIC in np.unique(df_rerun['TIC']):
    rows = df_rerun[df_rerun['TIC'] == TIC]
    df = df[df['TIC'] != TIC]
    df = df.append(rows)

df.to_csv('7.final.gauss_flare_out.csv')
