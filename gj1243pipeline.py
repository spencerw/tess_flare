import numpy as np
from flareFind import procFlares

path = '/astro/store/gradscratch/tmp/scw7/tessData/lightcurves/gj1243/'
filenames = np.genfromtxt('gj1243_files.txt', comments='#', dtype='str')
procFlares('gj1243', filenames, path, makePlots=True, clobberGP=True)
