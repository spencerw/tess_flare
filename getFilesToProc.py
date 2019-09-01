import numpy as np
import glob as gl
import pandas as pd
import sys

# Get all files, or search through param table and get files where p_res > 1e3?
first_pass = int(sys.argv[1])
path = '/gscratch/stf/scw7/tessFlares/all_sec/'

if first_pass:
    all_files = gl.glob(path+'*.fits')
    np.savetxt('files_to_proc.txt', all_files, delimiter='\n', fmt='%s')
else:
    path = '/gscratch/stf/scw7/tessFlares/all_sec/'
    df_param = pd.read_csv('all.gauss_param_out.csv')
    df_high_p = df_param[df_param['p_res'] > 1e3]
    files = df_high_p['file'].values
    print(str(len(files)) + ' files to rerun')
    all_files = []
    for f in files:
        all_files.append(path + f)
    np.savetxt('files_to_proc_rerun.txt', all_files, delimiter='\n', fmt='%s')
