import glob as gl
import numpy as np
import sys
import os

proc_num = int(sys.argv[1])
total_proc = int(sys.argv[2])

from flareFind import procFlares

all_files = np.genfromtxt('sec1to13_files.txt', dtype='str')

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
procFlares('1to13.' + str(proc_num), file_subset, '/gscratch/stf/scw7/tessFlares/sec1to13/', clobberGP=True)

print(str(proc_num) + ' finished!')
exit()
