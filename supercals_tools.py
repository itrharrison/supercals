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

def make_source_plot(config, bounds, clean_image, residual_image, model_stamp, stamp, image_to_measure, clean_psf_stamp, source, source_i, mod_e, theta):
  plt.close('all')
  nplots=2
  fig = plt.figure(1, figsize=(4.5, nplots*3.75))
  grid = AxesGrid(fig, 111,
                  nrows_ncols=(2,nplots),
                  axes_pad=0.0,
                  share_all=True,
                  label_mode='L')

  source_peak = clean_image[bounds].array.max() - residual_image[bounds].array.max()
  
  grid[0].imshow(clean_image[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[0].text(10,10,str(np.sum(clean_image[bounds].array)), color='white')
  grid[0].set_title('CLEAN')
  grid[0].axis('off')
  
  grid[0+nplots].plot(clean_image[bounds].array[clean_image[bounds].array.shape[0]/2,:])
  
  grid[1].imshow(residual_image[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  grid[1].set_title('Residual')
  grid[1].axis('off')
  
  grid[1+nplots].plot(residual_image[bounds].array[residual_image[bounds].array.shape[0]/2,:])
  
  '''
  
  grid[1].imshow(residual_image[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[1].text(10,10,str(np.sum(residual_image_gs[bounds].array)), color='white')
  grid[1].set_title('Residual')
  grid[1].axis('off')
  
  grid[2].imshow(model_stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[2].text(10,10,str(np.sum(stamp.array)), color='white')
  grid[2].set_title('Model')
  grid[2].axis('off')
  
  grid[3].imshow(stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[3].text(10,10,str(np.sum(stamp.array)), color='white')
  grid[3].set_title('Model+PSF')
  grid[3].axis('off')
  
  grid[4].imshow(image_to_measure.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[4].text(10,10,str(np.sum(image_to_measure.array)), color='white')
  grid[4].set_title('Model+PSF+Residual')
  grid[4].axis('off')
  
  grid[5].imshow(clean_psf_stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  grid[5].set_title('CLEAN PSF')
  grid[5].axis('off')
  
  #grid[6].imshow(dirty_psf_stamp, cmap='gnuplot2', interpolation='nearest', origin='lower')
  grid[6].set_title('Dirty PSF')
  grid[6].axis('off')
  
  grid[7].plot(clean_image[bounds].array[clean_image[bounds].array.shape[0]/2,:])
  grid[7].set_ylim([0,clean_image[bounds].array.max()])
  grid[7].set_xlim([0,166])
  
  grid[8].plot(residual_image[bounds].array[residual_image[bounds].array.shape[0]/2,:])
  grid[8].set_ylim([0,clean_image[bounds].array.max()])
  
  grid[9].plot(model_stamp.array[model_stamp.array.shape[0]/2,:])
  grid[9].set_ylim([0,clean_image[bounds].array.max()])
  
  grid[10].plot(stamp.array[stamp.array.shape[0]/2,:])
  grid[10].set_ylim([0,clean_image[bounds].array.max()])
  
  grid[11].plot(image_to_measure.array[image_to_measure.array.shape[0]/2,:])
  grid[11].set_ylim([0,clean_image[bounds].array.max()])
  
  clean_psf_oned = clean_psf_stamp.array[clean_psf_stamp.array.shape[0]/2,:]
  grid[12].plot(clean_psf_oned*(source_peak/clean_psf_oned.max()))
  
  #dirty_psf_oned = dirty_psf_stamp[dirty_psf_stamp.shape[0]/2,:]
  #grid[13].plot(dirty_psf_oned*(source_peak/dirty_psf_oned.max()))
  #grid[13].set_xlim([0,166])
  
  
  fig.subplots_adjust(hspace=0, wspace=0)
  pdb.set_trace()
  '''
  #plt.suptitle('{0} - {1}'.format(config.get('input', 'clean_image'), source['Source_id']))
  plt.savefig(config.get('output', 'output_plot_dir')+'/source_{0}_mode_{1}_rot_{2}.png'.format(source_i, mod_e, theta), dpi=300, bbox_inches='tight')

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
