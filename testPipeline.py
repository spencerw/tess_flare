import numpy as np
from flareFind import procFlares

path = 'test_files/'
filenames = np.genfromtxt(path + 'test_files.txt', comments='#', dtype='str')
procFlares('test', filenames, path, makePlots=True, clobberGP=False)
