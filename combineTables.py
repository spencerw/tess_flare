import pandas as pd
import glob as gl

# flare files
all_files = gl.glob('*_flare_out.csv')
prefix = all_files[0].split('.p')[0]

df_1 = pd.read_csv(all_files[0])

for f in all_files[1:]:
    df = pd.read_csv(f)
    df_1 = df_1.append(df)

df_1.to_csv(prefix+ '_flare_out.csv', index=False)

# param files
all_files = gl.glob('*_param_out.csv')
prefix = all_files[0].split('.p')[0]

df_1 = pd.read_csv(all_files[0])

for f in all_files[1:]:
    df = pd.read_csv(f)
    df_1 = df_1.append(df)

df_1.to_csv(prefix+ '_param_out.csv', index=False)
