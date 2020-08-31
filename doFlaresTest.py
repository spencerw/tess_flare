import glob as gl
from flareFind import procFlares
import numpy as np
import sys

all_files = np.genfromtxt('test_files.txt', dtype='str')

cpa_param = [3, 1, 3]

procFlares('test', all_files, 'test_files/', clobberGP=True)
