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
plt.close('all')

def make_source_plot(config, bounds, clean_image, residual_image, model_stamp, stamp, image_to_measure, clean_psf_stamp, source, source_i, mod_e, theta):
  plt.close('all')
  fig = plt.figure(1, figsize=(7*4.5, 2*3.75))
  ax2 = plt.subplot(271)
  plt.imshow(clean_image[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #plt.text(10,10,str(np.sum(clean_image[bounds].array)), color='white')
  plt.title('CLEAN')
  plt.axis('off')
  plt.subplot(272, sharey=ax2)
  plt.imshow(residual_image_gs[bounds].array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #plt.text(10,10,str(np.sum(residual_image_gs[bounds].array)), color='white')
  plt.title('Residual')
  plt.axis('off')
  plt.subplot(273, sharey=ax2)
  plt.imshow(model_stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #plt.text(10,10,str(np.sum(stamp.array)), color='white')
  plt.title('Model')
  plt.axis('off')
  plt.subplot(274, sharey=ax2)
  plt.imshow(stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #plt.text(10,10,str(np.sum(stamp.array)), color='white')
  plt.title('Model+PSF')
  plt.axis('off')
  plt.subplot(275, sharey=ax2)
  plt.imshow(image_to_measure.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  #plt.text(10,10,str(np.sum(image_to_measure.array)), color='white')
  plt.title('Model+PSF+Residual')
  plt.axis('off')
  plt.subplot(276, sharey=ax2)
  plt.imshow(clean_psf_stamp.array, cmap='gnuplot2', interpolation='nearest', origin='lower')
  plt.title('CLEAN PSF')
  plt.axis('off')
  plt.subplot(2,7,7, sharey=ax2)
  #plt.imshow(dirty_psf_stamp, cmap='gnuplot2', interpolation='nearest', origin='lower')
  plt.title('Dirty PSF')
  plt.axis('off')
  ax1 = plt.subplot(278)
  plt.plot(clean_image[bounds].array[clean_image[bounds].array.shape[0]/2,:])
  plt.ylim([0,clean_image[bounds].array.max()])
  plt.xlim([0,166])
  f = plt.gca()
  #f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(clean_image[bounds].array)), color='white')
  plt.subplot(279, sharex=ax1, sharey=ax1)
  plt.plot(residual_image[bounds].array[residual_image[bounds].array.shape[0]/2,:])
  plt.ylim([0,clean_image[bounds].array.max()])
  f = plt.gca()
  f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(residual_image_gs[bounds].array)), color='white')
  plt.subplot(2,7,10, sharex=ax1, sharey=ax1)
  plt.plot(model_stamp.array[model_stamp.array.shape[0]/2,:])
  plt.ylim([0,clean_image[bounds].array.max()])
  f = plt.gca()
  f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(stamp.array)), color='white')
  plt.subplot(2,7,11, sharex=ax1, sharey=ax1)
  plt.plot(stamp.array[stamp.array.shape[0]/2,:])
  plt.ylim([0,clean_image[bounds].array.max()])
  f = plt.gca()
  f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(stamp.array)), color='white')
  plt.subplot(2,7,12, sharex=ax1, sharey=ax1)
  plt.plot(image_to_measure.array[image_to_measure.array.shape[0]/2,:])
  plt.ylim([0,clean_image[bounds].array.max()])
  f = plt.gca()
  f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(image_to_measure.array)), color='white')
  plt.subplot(2,7,13, sharex=ax1, sharey=ax1)
  clean_psf_oned = clean_psf_stamp.array[clean_psf_stamp.array.shape[0]/2,:]
  plt.plot(clean_psf_oned*(source_peak/clean_psf_oned.max()))
  f = plt.gca()
  f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(clean_psf_stamp.array)), color='white')
  plt.subplot(2,7,14, sharex=ax1, sharey=ax1)
  #dirty_psf_oned = dirty_psf_stamp[dirty_psf_stamp.shape[0]/2,:]
  #plt.plot(dirty_psf_oned*(source_peak/dirty_psf_oned.max()))
  f = plt.gca()
  f.axes.yaxis.set_ticklabels([])
  f.axes.xaxis.set_ticklabels([])
  #plt.text(10,10,str(np.sum(dirty_psf_stamp)), color='white')
  plt.xlim([0,166])
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