import glob as gl
from flarePipeline import procFlares

sector = '7'
#path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/sec'+sector+'/'
path = '/astro/store/gradscratch/tmp/scw7/tessData/subset/' 

all_files = gl.glob(path+'*.fits')

procFlares(all_files, sector + '.gauss', makefig=False, clobber=True, smoothType='gauss_proc', progCounter=True)
