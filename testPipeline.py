import numpy as np
from flareFind import procFlares

path = 'test/'
filenames = np.genfromtxt(path + 'test_files.txt', comments='#', dtype='str')
procFlares('test', filenames, path, makePlots=False, clobberGP=True)
