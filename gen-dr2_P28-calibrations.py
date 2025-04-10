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

config = ConfigParser.ConfigParser()

base_dir = './data/jvla_dr1-mosaic_8orientations/'

part_cat_fnames = glob.glob(base_dir+'/jvla_dr1-mosaic-part*/mosaic-uncalibrated-shape-catalogue.fits')

mosaic_cat = Table()

for fname in part_cat_fnames:

  part_cat = Table.read(fname)
  mosaic_cat = vstack([mosaic_cat, part_cat])

mosaic_cat.write(base_dir+'/supercals/mosaic-uncalibrated-shape-catalogue.fits', overwrite=True)

config.add_section('catalogues')
config.set('catalogues', 'base_dir', base_dir)
config.set('catalogues', 'wl_catalogue', base_dir+'/cats/jvla_dr1.SIN_split.concat.srl.masked.pub.fits')
config.set('catalogues', 'supercals_directory', base_dir+'/supercals/')
config.set('catalogues', 'supercals_catalogue', base_dir+'/supercals/mosaic-uncalibrated-shape-catalogue.fits')

calibrate_supercals_catalogue(config, doplots=True)

pointing_cat_fname = base_dir+'/cats/jvla_dr1.SIN_split.concat.srl.masked.pub.supercals-calibrated.fits'
pointing_cat = Table.read(pointing_cat_fname)

gold_cat = pointing_cat[pointing_cat['radius_im3shape'] > 5]

print(len(gold_cat))

make_cat_calibration_plots(gold_cat, base_dir='./data/jvla_dr1-mosaic_8orientations/', name='gold')

gold_cat.write(base_dir+'/cats/jvla_dr1.SIN_split.concat.srl.masked.pub.supercals-calibrated.gold.fits', overwrite=True)