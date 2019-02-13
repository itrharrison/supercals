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

pointing_dirs = [
                 '/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla-J28/1times-size/',
                 '/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla-J28/3times-size/',
                 '/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla-J28/5times-size/',
                 '/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla-J28/gaussian-1times-size/',
                 '/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla-J28/gaussian-3times-size/',
                 '/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla-J28/gaussian-5times-size/',
                 ]

all_pointings_cat = Table()

for ptg in pointing_dirs:

  print(ptg)

  config = ConfigParser.ConfigParser()

  ptg_name = (ptg.split('/')[-2]).replace('-size', '')

  config.add_section('catalogues')
  config.set('catalogues', 'base_dir', ptg)
  config.set('catalogues', 'wl_catalogue', ptg+'/cats/J28.tclean.image.tt0_split_000.srl.resolved.fits')
  config.set('catalogues', 'supercals_directory', ptg+'/supercals/')
  config.set('catalogues', 'supercals_catalogue', ptg+'/supercals/{0}-uncalibrated-shape-catalogue.fits'.format(ptg_name))

  calibrate_supercals_catalogue(config, truth_cat_fname='/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla/truthcats/level2-jvla-indiv-J28_truthcat.fits')

  #pointing_cat_fname = ptg+'/cats/{0}.tclean.image.tt0_split_000.srl.resolved.supercals-calibrated.fits'.format(ptg_name)

  #pointing_cat = Table.read(pointing_cat_fname)

  #all_pointings_cat = vstack([all_pointings_cat, pointing_cat])

#all_pointings_cat.write('data/level2-jvla/level2-jvla-indiv.supercals-calibrated.fits', overwrite=True)

#make_cat_calibration_plots(all_pointings_cat, base_dir='./data/level2-jvla/', name='all_pointings')