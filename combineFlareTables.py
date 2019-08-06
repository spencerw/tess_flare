import pandas as pd
import glob as gl

all_files = gl.glob('*_flare_out.csv')
prefix = all_files[0].split('.p')[0]

data = []

for f in all_files:
    df = pd.read_csv(f)
    data.append(df)

data = pd.concat(data)
data.to_csv(prefix+ '_flare_out.csv')
