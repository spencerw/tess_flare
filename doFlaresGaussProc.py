import glob as gl
from flarePipeline import procFlaresGP
import numpy as np
import sys

proc_num = int(sys.argv[1])
total_proc = int(sys.argv[2])

sector = '7.rerun'
all_files = np.genfromtxt('files_to_rerun.txt', dtype='str')

chunk_size = len(all_files)//total_proc
i1 = chunk_size*proc_num
i2 = chunk_size*(proc_num + 1)

file_subset = all_files[i1:i2]
print('Process ' + str(i1) + ' to ' + str(i2))
print(str(len(file_subset)) + ' total files')

procFlaresGP(file_subset, sector + '.gauss.p' + str(proc_num), makefig=False, clobberPlots=True, clobberGP=True, writeLog=True, debug=True, gpInterval=1)

print(str(proc_num) + ' finished!')
exit()
