import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import glob
import os
import configparser

from astropy.table import Table, vstack

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

from calibrate_catalogue import *
from bias_surface import *

template = open('inis/level2-jvla-J28-sizes-template.ini').read()

run_list = [
            '1times-size-1times-flux',
            '2times-size-2times-flux',
            '3times-size-5times-flux',
            '3times-size-100times-flux',
            #'./exp-1times-size-1times-flux/',
            #'./simple-psf/'
            ]


for run in run_list:

  ini_filename = 'inis/level2-jvla-J28-{0}.ini'.format(run)

  ini_file = template.format(ptg=run)
  open(ini_filename, 'w').write(ini_file)
 