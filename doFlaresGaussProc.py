import glob as gl
from flarePipeline import procFlaresGP

sector = '7'
path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/sec'+sector+'/'
#path = '/astro/store/gradscratch/tmp/scw7/tessData/subset/' 

all_files = gl.glob(path+'*.fits')

procFlaresGP(all_files, sector + '.gauss', makefig=True, clobberPlots=True, clobberGP=True, writeLog=True)
