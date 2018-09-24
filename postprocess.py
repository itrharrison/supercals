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

def runCalibration(pointing, config):
  '''
  Creates a calibrated cat for a pointing.
  Could be a method in a pointing class, but really?

  '''
  uncalibrated_cat = Table.read(pointing['uncalibrated_cat_fname'])
  
  if len(pointing_uncalibrated_cat) < 1:
    print('No detected sources in {0} pointing!',format(pointing['name']))
    return -1

  pointing_calibrated_cat = copy.copy(pointing_uncalibrated_cat)
  pointing_calibrated_cat['m_e1'] = np.nan
  pointing_calibrated_cat['m_e2'] = np.nan
  pointing_calibrated_cat['c_e1'] = np.nan
  pointing_calibrated_cat['c_e2'] = np.nan
  pointing_calibrated_cat['e1_calibrated'] = np.nan
  pointing_calibrated_cat['e2_calibrated'] = np.nan

  for src in pointing_calibrated_cat:

    calibration_cat_fname = 
    calibration_cat = Table.read(calibration_cat_fname, format='fits')
    fit = make_m_and_c(calibration_cat)
    src['m_e1'] = fit['m_e1']
    src['m_e2'] = fit['m_e2']
    src['c_e1'] = fit['c_e1']
    src['c_e2'] = fit['c_e2']
    src['e1_calibrated'] = (src['e1_obs'] - fit['c_e1'])/(1.e0 + fit['m_e1']) 
    src['e2_calibrated'] = (src['e2_obs'] - fit['c_e2'])/(1.e0 + fit['m_e2'])

    sigma2_e1 = 0.e0
    sigma2_e2 = 0.e0
    src['sigma2_e1_calibrated'] = (sigma2_e1 + fit['sigma2_c_e1'] + fit['sigma2_m_e1']*((fit['c_e1'] - src['e1_obs'])/(1.e0 + fit['m_e1']))**2.)/(1.e0 + fit['m_e1'])**2.
    src['sigma2_e2_calibrated'] = (sigma2_e2 + fit['sigma2_c_e2'] + fit['sigma2_m_e2']*((fit['c_e2'] - src['e2_obs'])/(1.e0 + fit['m_e2']))**2.)/(1.e0 + fit['m_e2'])**2.

  pointing_calibrated_cat.write(pointing['uncalibrated_cat_fname'].replace('uncalibrated','calibrated'), overwrite=True)


def runAverageCatalogues(config):

  mosaic_cat = Table.read()

  check_source_ids(mosaic_cat)

  for src in mosaic_cat:

    src_e1_cal_arr = np.array([])
    src_e2_cal_arr = np.array([])
    src_e1_uncal_arr = np.array([])
    src_e2_uncal_arr = np.array([])
    src_snr_arr = np.array([])
    src_pointings_arr = np.array([])

    for pointing in pointings:

      if not os.path.exists(pointing['uncalibrated_cat_fname'].replace('uncalibrated','calibrated'), overwrite=True):
        print('No calibrations found in pointing {0}'.format(pointing['name']))
        continue

      calibrated_cat = Table.read(pointing['uncalibrated_cat_fname'].replace('uncalibrated','calibrated'))
      list_of_measurements = calibrated_cat[calibrated_cat['Source_id']==src['Source_id']]

      if len(list_of_measurements) < 1:
        print('No calibrations found for {0} in pointing {1}'.format(src['Source_id'], pointing['name']))
        continue

      if not isinstance(list_of_measurements, (list,)):
        list_of_measurements = [list_of_measurements]

      for measurement in list_of_measurements:

        src_e1_cal_arr = np.append([src_e1_cal_arr, measurement['e1_calibrated']])
        src_e2_cal_arr = np.append([src_e2_cal_arr, measurement['e2_calibrated']])
        src_e1_uncal_arr = np.append([src_e1_uncal_arr, measurement['e1_obs']])
        src_e2_uncal_arr = np.append([src_e2_uncal_arr, measurement['e2_obs']])
        src_pointings_arr = np.append([src_pointings_arr, pointing.split('/')[2]])
        src_snr_arr = np.append([src_snr_arr, measurement['snr'].data[0]])

    src_mode_cal_arr = np.sqrt(src_e1_cal_arr**2 + src_e2_cal_arr**2.).flatten()
    src_mode_uncal_arr = np.sqrt(src_e1_uncal_arr**2 + src_e2_uncal_arr**2.).flatten()

    src['e1_calibrated'] = np.mean(src_e1_cal_arr) # consider working this with nanmean
    src['e2_calibrated'] = np.mean(src_e2_cal_arr)
    src['e1_uncalibrated'] = np.mean(src_e1_uncal_arr)
    src['e2_uncalibrated'] = np.mean(src_e2_uncal_arr)


'''
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
'''