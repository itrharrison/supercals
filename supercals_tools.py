import pdb
import ConfigParser
import numpy as np
import sys
import os
import time
import cPickle as pickle
import pdb
from math import ceil

from astropy.io import fits
from astropy.table import Table, join, Column
import astropy.table as tb
from astropy import wcs

import galsim

sys.path.append('../simuclass')

from skymodel.skymodel_tools import setup_wcs

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid
plt.close('all')

def add_source_subplot(grid, i_plot, n_plots, image, label, global_norm=False):

  grid[i_plot].imshow(image, cmap='gnuplot2', interpolation='nearest', origin='lower')
  grid[i_plot].set_xlim([0,image.shape[0]])
  grid[i_plot].set_ylim([0,image.shape[1]])
  grid[i_plot].axis('off')

  grid[i_plot+n_plots].plot(image[:,image.shape[1]/2],'k-')
  grid[i_plot+n_plots].set_xlim([0,image.shape[0]])
  if bool(global_norm:)
    grid[i_plot+n_plots].set_ylim([0,global_norm])
  grid[i_plot+n_plots].axis('off')

  grid[i_plot].set_title(label, size='x-small')  

def make_source_plot(config, bounds, clean_image, residual_image, model_stamp, obsgal_stamp, image_to_measure, clean_psf_stamp, dirty_psf_stamp, source, source_i, mod_e, theta):
  plt.close('all')
  nplots=2
  fig = plt.figure(1, figsize=(4.5, nplots*3.75))
  grid = AxesGrid(fig, 111,
                  nrows_ncols=(2,nplots),
                  axes_pad=0.0,
                  share_all=False,
                  label_mode='L')

  source_peak = clean_image[bounds].array.max() - residual_image[bounds].array.max()
  
  add_source_subplot(grid, 0, clean_image[bounds].array, 'CLEAN', global_norm=source_peak)
  add_source_subplot(grid, 1, residual_image[bounds].array, 'Residual', global_norm=source_peak)
  add_source_subplot(grid, 2, model_stamp.array, 'Model', global_norm=source_peak)
  add_source_subplot(grid, 3, obsgal_stamp.array, 'Model+PSF', global_norm=source_peak)
  add_source_subplot(grid, 4, image_to_measure.array, 'Model+PSF+Residual', global_norm=source_peak)
  add_source_subplot(grid, 5, clean_psf_stamp.array, 'CLEAN PSF', global_norm=source_peak)
  add_source_subplot(grid, 6, dirty_psf_stamp.array, 'Dirty PSF', global_norm=source_peak)
  
  plt.suptitle('{0} \n {1}'.format(config.get('input', 'clean_image').split['/'][-1], source['Source_id']), size='x-small')
  plt.savefig(config.get('output', 'output_plot_dir')+'/{0}_mode_{1}_rot_{2}.png'.format(source['Source_id'], mod_e, theta), dpi=300, bbox_inches='tight')

def source_in_pointing(source, w_twod, npix):

  x, y = w_twod.wcs_world2pix(source['RA'], source['DEC'], 0,)
  if (0 <= x <= npix[0]) and (0 <= y <= npix[1]):
    return True
  else:
    return False

def get_stamp_size(source, pix_scale):

  stamp_size = 10.* (source['Maj']*galsim.degrees / galsim.arcsec) / (pix_scale / galsim.arcsec)
  stamp_size = ceil(stamp_size/2.) *2  
  
  return int(stamp_size)

def process_im3shape_result(config, source, result, best_fit):
  
  from py3shape.analyze import analyze, count_varied_params
  from py3shape.options import Options
  
  options = Options(config.get('im3shape', 'ini_file'))
  
  result = result.as_dict(0, count_varied_params(options))
  radius = result[0]['radius']
  # convert between im3shape and galsim ellipticity definitions
  g1_obs = result[0]['e1']
  g2_obs = result[0]['e2']

  g_obs = np.sqrt(g1_obs**2. + g2_obs**2.)
  q_obs = (1. - g_obs)/(1. + g_obs)

  e_obs = (1. - q_obs**2.)/(1. + q_obs**2.)

  e1_obs = g1_obs*e_obs/g_obs
  e2_obs = g2_obs*e_obs/g_obs

  snr = result[0]['snr']
  likelihood = result[0]['likelihood']
  disc_A = result[0]['disc_A']
  disc_flux = result[0]['disc_flux']
  
  result[0]['g1_obs'] = g1_obs
  result[0]['g2_obs'] = g2_obs
  
  result[0]['e1_obs'] = e1_obs
  result[0]['e2_obs'] = e2_obs
  
  result[0]['q_obs'] = q_obs
  
  retvar = result[0]

  return retvar
