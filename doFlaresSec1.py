import glob as gl
from flarePipeline import procFlaresGP
import numpy as np
import sys

proc_num = int(sys.argv[1])
total_proc = int(sys.argv[2])

all_files = np.genfromtxt('sec1_files.txt', dtype='str')

chunk_size = len(all_files)//total_proc + 1
i1 = chunk_size*proc_num
i2 = chunk_size*(proc_num + 1)

if i2 > len(all_files):
    i2 = len(all_files)

file_subset = all_files[i1:i2]
print('Process ' + str(i1) + ' to ' + str(i2))
print(str(len(file_subset)) + ' total files')
print(file_subset[0] + ' to ' + file_subset[-1])

cpa_param = [3, 1, 3]
procFlaresGP(file_subset, 'sec1.p' + str(proc_num), cpa_param, makefig=False, clobberPlots=True, clobberGP=False, writeLog=True, debug=True, gpInterval=1)

print(str(proc_num) + ' finished!')
exit()