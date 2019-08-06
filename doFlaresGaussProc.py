import glob as gl
from flarePipeline import procFlaresGP
import sys

proc_num = int(sys.argv[1])
total_proc = int(sys.argv[2])

sector = '7'
#path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/sec'+sector+'/'
path = '/gscratch/stf/scw7/tessFlares/sec7/'
all_files = gl.glob(path+'*.fits')

chunk_size = len(all_files)//total_proc
i1 = chunk_size*proc_num
i2 = chunk_size*(proc_num + 1)

file_subset = all_files[i1:i2]
print('Process ' + str(i1) + ' to ' + str(i2))
print(str(len(file_subset)) + ' total files')

procFlaresGP(file_subset, sector + '.gauss.p' + str(proc_num), makefig=False, clobberPlots=True, clobberGP=True, writeLog=True, debug=True)

print(str(proc_num) + ' finished!')
exit()
