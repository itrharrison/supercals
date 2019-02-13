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
                 #'data/level2-jvla/level2-jvla-indiv-J22',
                 #'data/level2-jvla/level2-jvla-indiv-J32',
                 #'data/level2-jvla/level2-jvla-indiv-K25',
                 #'data/level2-jvla/level2-jvla-indiv-K27',
                 #'data/level2-jvla/level2-jvla-indiv-L20',
                 #'data/level2-jvla/level2-jvla-indiv-M23',
                 #'data/level2-jvla/level2-jvla-indiv-N22',
                 # 'data/level2-jvla/level2-jvla-indiv-N24',
                 # 'data/level2-jvla/level2-jvla-indiv-N26',
                 # 'data/level2-jvla/level2-jvla-indiv-N28',
                 # 'data/level2-jvla/level2-jvla-indiv-N30',
                 # 'data/level2-jvla/level2-jvla-indiv-N32',
                 # 'data/level2-jvla/level2-jvla-indiv-N34',
                 # 'data/level2-jvla/level2-jvla-indiv-O21',
                 # 'data/level2-jvla/level2-jvla-indiv-O23',
                 # 'data/level2-jvla/level2-jvla-indiv-O25',
                 # 'data/level2-jvla/level2-jvla-indiv-O27',
                 # 'data/level2-jvla/level2-jvla-indiv-O29',
                 # 'data/level2-jvla/level2-jvla-indiv-O31',
                 # 'data/level2-jvla/level2-jvla-indiv-O33',
                 # 'data/level2-jvla/level2-jvla-indiv-P22',
                 # 'data/level2-jvla/level2-jvla-indiv-P24',
                 # 'data/level2-jvla/level2-jvla-indiv-P26',
                 # 'data/level2-jvla/level2-jvla-indiv-P28'
                 'data/level2-jvla/level2-jvla-indiv-J28'
                 ]

all_pointings_cat = Table()

for ptg in pointing_dirs:

  print(ptg)

  config = ConfigParser.ConfigParser()

  ptg_name = ptg[-3:]

  config.add_section('catalogues')
  config.set('catalogues', 'base_dir', ptg)
  config.set('catalogues', 'wl_catalogue', ptg+'/cats/{0}.tclean.image.tt0_split_000.srl.resolved.fits'.format(ptg_name))
  config.set('catalogues', 'supercals_directory', ptg+'/supercals/')
  config.set('catalogues', 'supercals_catalogue', './data/level2-jvla/level2-jvla-indiv-{0}/supercals/{0}-uncalibrated-shape-catalogue.fits'.format(ptg_name))

  calibrate_supercals_catalogue(config, truth_cat_fname='/Users/harrison/Dropbox/code_mcr/supercals/data/level2-jvla/truthcats/level2-jvla-indiv-{0}_truthcat.fits'.format(ptg_name))

  pointing_cat_fname = ptg+'/cats/{0}.tclean.image.tt0_split_000.srl.resolved.supercals-calibrated.fits'.format(ptg_name)

  pointing_cat = Table.read(pointing_cat_fname)

  all_pointings_cat = vstack([all_pointings_cat, pointing_cat])

all_pointings_cat.write('data/level2-jvla/level2-jvla-indiv.supercals-calibrated.fits', overwrite=True)

make_cat_calibration_plots(all_pointings_cat, base_dir='./data/level2-jvla/', name='all_pointings')