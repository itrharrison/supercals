import pdb
try:
  import configparser as ConfigParser
except:
  import ConfigParser
import numpy as np
import sys
import os
import time
import pickle
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

  #grid[i_plot].plot(image[:,image.shape[1]/2],'k-')
  #grid[i_plot+n_plots].set_xlim([0,image.shape[0]])
  #if bool(global_norm):
  #grid[i_plot+n_plots].set_ylim([0,global_norm])
  #grid[i_plot+n_plots].axis('off')
  grid[i_plot].set_title(label, size=3)  

def make_source_plot(config, bounds, mosaic_image_array, clean_image, residual_image, model_stamp, obsgal_stamp, image_to_measure, clean_psf_stamp, dirty_psf_stamp, model_image_stamp, source, source_i, mod_e, theta, clean_header):
  plt.close('all')
  nplots=9
  fig = plt.figure(1, figsize=(4.5, nplots*3.75))
  grid = AxesGrid(fig, 111,
                  nrows_ncols=(1,nplots),
                  axes_pad=0.0,
                  share_all=False,
                  label_mode='L')

  offset_dist = np.sqrt((clean_header['CRVAL1']-source['RA'])**2. + (clean_header['CRVAL2']-source['DEC'])**2.)
  source_peak = image_to_measure.array.max()
  add_source_subplot(grid, 0, nplots, mosaic_image_array, 'Mosaic', global_norm=source_peak)
  add_source_subplot(grid, 1, nplots, clean_image[bounds].array, 'CLEAN', global_norm=source_peak)
  add_source_subplot(grid, 2, nplots, residual_image[bounds].array, 'Residual\nRMS: {0:.2e}\nOffset: {1:.3f}'.format(np.sqrt(np.var(residual_image[bounds].array)), offset_dist), global_norm=source_peak)
  add_source_subplot(grid, 3, nplots, model_stamp.array, 'Model', global_norm=source_peak)
  add_source_subplot(grid, 4, nplots, obsgal_stamp.array, 'Model+PSF', global_norm=source_peak)
  add_source_subplot(grid, 5, nplots, image_to_measure.array, 'Model+PSF+Residual', global_norm=source_peak)
  add_source_subplot(grid, 6, nplots, clean_psf_stamp.array, 'CLEAN PSF', global_norm=source_peak)
  add_source_subplot(grid, 7, nplots, dirty_psf_stamp, 'Dirty PSF', global_norm=source_peak)
  add_source_subplot(grid, 8, nplots, model_image_stamp, 'CLEAN Comps.\nN={0}\n{1:.2e}'.format(np.sum(model_image_stamp!=0), np.sum(model_image_stamp)))
  os.system('echo \'{0},{1:.3f},{2:.2e},{3},{4:.2e}\' >> /home/harrison/offsets.txt'.format(source['Source_id'], offset_dist, np.sqrt(np.var(residual_image[bounds].array)), np.sum(model_image_stamp!=0), np.sum(model_image_stamp)))
  #plt.suptitle('{0} \n {1}'.format(config.get('input', 'clean_image').split('/')[-1], source['Source_id']), size=3)
  plt.savefig(config.get('output', 'output_plot_dir')+'/{0}_mode_{1}_rot_{2}.png'.format(source['Source_id'], mod_e, theta), dpi=300, bbox_inches='tight')

def source_in_pointing(source, w_twod, npix):

  x, y = w_twod.wcs_world2pix(source['RA'], source['DEC'], 0,)
  if (0 <= x <= npix[0]) and (0 <= y <= npix[1]):
    return True
  else:
    return False

def get_stamp_size(source, pix_scale):

  stamp_size = 20.* (source['Maj']*galsim.degrees / galsim.arcsec) / (pix_scale / galsim.arcsec)
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
