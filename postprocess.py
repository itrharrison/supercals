import pdb
import ConfigParser
import numpy as np
import sys
import os
import time
import cPickle as pickle
import pdb
from math import ceil
import glob

from astropy.io import fits
from astropy.table import Table, join, Column
import astropy.table as tb
from astropy import wcs

sys.path.append('../simuclass')

from skymodel.skymodel_tools import setup_wcs
from postprocess_tools import calculate_corrected_ellipticity, make_m_and_c, make_ein_eout_plots, make_calib_surface_plots

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.close('all')

def runCalibration(config):
  # need to think about this
  # do we want to used averaged corrections on averaged measurements?
  # is this equivalent to averaging after correction (probably)?
  #for source in wl_cat:
  #  make_m_and_c(source, config)
  #  calculate_corrected_ellipticity(source_in_pointing, config)
  #  make_calib_surface_plots(config)

  # generate a PER POINTING CALIBRATED CATALOGUE
  uncalibrated_cat_fname = config.get('output', 'output_cat_dir')+'/{0}-uncalibrated-shape-catalogue.fits'.format(config.get('input', 'pointing_name'))
  calibration_cat_fname = config.get('output', 'output_cat_dir')+'/{0}_supercals.fits'.format(source['Source_id'])
  calibrated_cat_fname = config.get('output', 'output_cat_dir')+'/{0}-calibrated-shape-catalogue.fits'.format(config.get('input', 'pointing_name'))

  uncalibrated_cat = Table.read(uncalibrated_cat_fname)
  calibrated_cat = uncalibrated_cat
  calibrated_cat['e1_calibrated'] = []
  calibrated_cat['sigma2_e1_calibrated'] = []
  calibrated_cat['e2_calibrated'] = []
  calibrated_cat['sigma2_e1_calibrated'] = []

  for source_i, source in enumerate(uncalibrated_cat):
    
    calibration_fit = make_m_and_c(source, config)

    # im3shape is a maximum likelihood code and doesn't give any output about the curvature...
    sigma2_e1 = 0.e0
    sigma2_e2 = 0.e0
    source['e1_calibrated'] = (source['e1_obs'] - c_e1)/(1.e0 + m_e1)
    source['e2_calibrated'] = (source['e2_obs'] - c_e2)/(1.e0 + m_e2)
    source['sigma2_e1_corrected'] = (sigma2_e1 + sigma2_c_e1 + sigma2_m_e1*((c_e1 - source['e1_obs'])/(1.e0 + m_e1))**2.)/(1.e0 + m_e1)**2.
    source['sigma2_e2_corrected'] = (sigma2_e2 + sigma2_c_e2 + sigma2_m_e2*((c_e2 - source['e2_obs'])/(1.e0 + m_e2))**2.)/(1.e0 + m_e2)**2.
    #make_calib_surface_plots(config)

  calibrated_cat.write(calibrated_cat_fname, format='fits')

def runCreateCatalogue(config):
<<<<<<< HEAD
  # from per pointing calibrated catalogue, take averages to get calibrated source shape measurements
  cat = Table.read(config.get('input', 'catalogue'), format='fits')

  calibration_cat_fname = config.get('output', 'output_cat_dir')+'/{0}_supercals.fits'.format(source['Source_id'])

  pointing_list = config.get('survey', 'pointing_list').split(',')


  for pointing in pointing_list:

    calibration_cat_fname = config.get('output', 'output_cat_dir')+'/{0}-calibrated-shape-catalogue.fits'.format(config.get('input', 'pointing_name'))
    calibration_cat = Table.read(calibration_cat_fname)

    for source_i, source in enumerate(cat):
      
      # todo: work out what kind of data structure will actually let you do what you want (pd?)
      
      if source['Source_id'] not in calibration_cat['Source_id']:
        continue

      source_calibration = calibration_cat[calibration_cat['Source_id']==source['Source_id']]

      cat['Source_id'==source['Source_id']]['calibrated_e1_list'].append(source_calibration['e1_calibrated'])
      cat['Source_id'==source['Source_id']]['calibrated_e2_list'].append(source_calibration['e2_calibrated'])
      cat['Source_id'==source['Source_id']]['calibrated_sigma2_e1_list'].append(source_calibration['sigma2_e1_calibrated'])
      cat['Source_id'==source['Source_id']]['calibrated_sigma2_e2_list'].append(source_calibration['sigma2_e2_calibrated'])


  for source_i, source in enumerate(cat)

    cat['Source_id'==source['Source_id']]['calibrated_e1'] = np.average(source['calibrated_e1_list'], weights=)
    cat['Source_id'==source['Source_id']]['calibrated_e2'] = np.average(source['calibrated_e2_list'], weights=)

  cat.write(config.get('input', 'catalogue').rstrip('.fits')+'-withcal.fits', format='fits', overwrite=True)



=======
  # from per pointing calibrated catalogue, take averages to get calibrated source shape measurement!
  return 0
>>>>>>> 5f25cb4a3b881b40ff10220560b49217efc7efdb
