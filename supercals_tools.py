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

  fig = plt.figure(1, figsize=(7*4.5, 2*3.75))
  grid = AxesGrid(fig, 111,
                  nrows_ncols=(2,7),
                  axes_pad=0.0,
                  share_all=True,
                  label_mode='L')


  grid[0].imshow(clean_image[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[0].text(10,10,str(np.sum(clean_image[bounds].array)), color='white')
  grid[0].title('CLEAN')
  grid[0].axis('off')
  grid[0].subplot(272, sharey=ax2)
  grid[0].imshow(residual_image_gs[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[0].text(10,10,str(np.sum(residual_image_gs[bounds].array)), color='white')
  grid[0].title('Residual')
  grid[0].axis('off')
  
  grid[1].imshow(model_stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[1].text(10,10,str(np.sum(stamp.array)), color='white')
  grid[1].title('Model')
  grid[1].axis('off')
  
  grid[2].imshow(stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[2].text(10,10,str(np.sum(stamp.array)), color='white')
  grid[2].title('Model+PSF')
  grid[2].axis('off')
  
  grid[3].imshow(image_to_measure.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #grid[3].text(10,10,str(np.sum(image_to_measure.array)), color='white')
  grid[3].title('Model+PSF+Residual')
  grid[3].axis('off')
  
  grid[4].imshow(clean_psf_stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  grid[4].title('CLEAN PSF')
  grid[4].axis('off')
  
  #grid[5].imshow(dirty_psf_stamp, cmap='gnuplot2', interpolation='nearest', origin='lower')
  grid[5].title('Dirty PSF')
  grid[5].axis('off')
  
  grid[6].plot(clean_image[bounds].array[clean_image[bounds].array.shape[0]/2,:])
  grid[6].ylim([0,clean_image[bounds].array.max()])
  grid[6].xlim([0,166])
  f = grid[6].gca()
  #f.axes.yaxis.set_ticklabels([6])
  #f.axes.xaxis.set_ticklabels([6])
  #grid[6].text(10,10,str(np.sum(clean_image[bounds].array)), color='white')
  
  grid[7].plot(residual_image[bounds].array[residual_image[bounds].array.shape[0]/2,:])
  grid[7].ylim([0,clean_image[bounds].array.max()])
  f = grid[7].gca()
  #f.axes.yaxis.set_ticklabels([7])
  #f.axes.xaxis.set_ticklabels([7])
  #grid[7].text(10,10,str(np.sum(residual_image_gs[bounds].array)), color='white')
  
  #grid[8].subplot(2,7,10, sharex=ax1, sharey=ax1)
  grid[8].plot(model_stamp.array[model_stamp.array.shape[0]/2,:])
  grid[8].ylim([0,clean_image[bounds].array.max()])
  #f = grid[8].gca()
  #f.axes.yaxis.set_ticklabels([8])
  #f.axes.xaxis.set_ticklabels([8])
  #grid[].text(10,10,str(np.sum(stamp.array)), color='white')
  
  #grid[9].subplot(2,7,11, sharex=ax1, sharey=ax1)
  grid[9].plot(stamp.array[stamp.array.shape[0]/2,:])
  grid[9].ylim([0,clean_image[bounds].array.max()])
  #f = grid[9].gca()
  #f.axes.yaxis.set_ticklabels([9])
  #f.axes.xaxis.set_ticklabels([9])
  #grid[].text(10,10,str(np.sum(stamp.array)), color='white')
  
  #grid[10].subplot(2,7,12, sharex=ax1, sharey=ax1)
  grid[10].plot(image_to_measure.array[image_to_measure.array.shape[0]/2,:])
  grid[10].ylim([0,clean_image[bounds].array.max()])
  #f = grid[10].gca()
  #f.axes.yaxis.set_ticklabels([10])
  #f.axes.xaxis.set_ticklabels([10])
  #grid[10].text(10,10,str(np.sum(image_to_measure.array)), color='white')
  
  #grid[11].subplot(2,7,13, sharex=ax1, sharey=ax1)
  clean_psf_oned = clean_psf_stamp.array[clean_psf_stamp.array.shape[0]/2,:]
  grid[11].plot(clean_psf_oned*(source_peak/clean_psf_oned.max()))
  #f = grid[11].gca()
  #f.axes.yaxis.set_ticklabels([11])
  #f.axes.xaxis.set_ticklabels([11])
  #grid[11].text(10,10,str(np.sum(clean_psf_stamp.array)), color='white')
  
  #grid[13].subplot(2,7,14, sharex=ax1, sharey=ax1)
  #dirty_psf_oned = dirty_psf_stamp[dirty_psf_stamp.shape[0]/2,:]
  #grid[13].plot(dirty_psf_oned*(source_peak/dirty_psf_oned.max()))
  #f = grid[13].gca()
  #f.axes.yaxis.set_ticklabels([13])
  #f.axes.xaxis.set_ticklabels([13])
  #grid[13].text(10,10,str(np.sum(dirty_psf_stamp)), color='white')
  #grid[13].xlim([0,166])
  
  plt.suptitle(source['Source_id'])
  fig.subplots_adjust(hspace=0, wspace=0)

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