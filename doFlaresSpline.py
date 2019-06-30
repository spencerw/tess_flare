import glob as gl
from flarePipeline import procFlares

sector = '7'
path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/sec'+sector+'/'

all_files = gl.glob(path+'*.fits')

procFlares(all_files, sector + '.spline', makefig=False, clobber=True, doSpline=True)
