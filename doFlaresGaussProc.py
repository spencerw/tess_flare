import glob as gl
from flarePipeline import procFlaresGP
import numpy as np
import sys

proc_num = int(sys.argv[1])
total_proc = int(sys.argv[2])

sector = 'all'
all_files = np.genfromtxt('files_to_proc.txt', dtype='str')

chunk_size = len(all_files)//total_proc + 1
i1 = chunk_size*proc_num
i2 = chunk_size*(proc_num + 1)

if i2 > len(all_files):
    i2 = len(all_files)

file_subset = all_files[i1:i2]
print('Process ' + str(i1) + ' to ' + str(i2))
print(str(len(file_subset)) + ' total files')
print(file_subset[0] + ' to ' + file_subset[-1])

procFlaresGP(file_subset, sector + '.gauss.p' + str(proc_num), makefig=False, clobberPlots=True, clobberGP=True, writeLog=True, debug=True, gpInterval=15)

print(str(proc_num) + ' finished!')
exit()
