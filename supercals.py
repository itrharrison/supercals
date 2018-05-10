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
from supercals_tools import make_source_plot, source_in_pointing, get_stamp_size, process_im3shape_result

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
plt.close('all')

image_output_columns = ('Source_id', 'g1_obs', 'g2_obs', 'e1_obs', 'e2_obs', 'radius', 'snr', 'likelihood', 'disc_A', 'disc_flux')  
calibration_output_columns = names=('Source_id', 'mod_g', 'theta_g', 'mod_e', 'theta_e', 'g1_inp', 'g2_inp', 'e1_inp', 'e2_inp', 'g1_obs', 'g2_obs', 'e1_obs', 'e2_obs', 'radius', 'snr', 'likelihood', 'disc_A', 'disc_flux')

big_fft_params = galsim.GSParams(maximum_fft_size=81488)

def runSuperCal(config):
  
  print('######################################')
  print('       START POINTING {0}       '.format(config.get('input', 'pointing_name')))
  print('######################################')

  # set up output names
  image_output_cat = Table(names=image_output_columns,  dtype=['S27']+(len(image_output_columns)-1)*[float])
  image_output_cat_fname = config.get('output', 'output_cat_dir')+'/uncalibrated-shape-catalogue.fits'
  
  if not os.path.exists(config.get('output', 'output_cat_dir')):
    os.makedirs(config.get('output', 'output_cat_dir'))
  if not os.path.exists(config.get('output', 'output_plot_dir')):
    os.makedirs(config.get('output', 'output_plot_dir'))
  
  # load im3shape
  im3dir = config.get('im3shape', 'install_directory')

  sys.path.append(im3dir)

  from py3shape.analyze import analyze, count_varied_params
  from py3shape.options import Options
  
  options = Options(config.get('im3shape', 'ini_file'))

  # set up ring
  n_shears = config.getint('ring','n_shears') # 1
  n_orientations = config.getint('ring','n_orientations') # 8
  n_ellipticities = config.getint('ring','n_ellipticities') # 15
  
  n_evals = n_shears*n_orientations*n_ellipticities

  orientations = np.linspace(0.e0, np.pi, n_orientations+1)
  orientations = orientations[:-1]
  ellipticities = np.linspace(0.e0, 0.7, n_ellipticities)
  ellipticities = ellipticities[::-1]
  shears = np.linspace(0.e0, 0.7, n_shears)
  shear_theta = 0.e0
  
  # load catalogue positions
  cat = Table.read(config.get('input', 'catalogue'), format='fits')
  
  # ToDo: make this unnecessary.
  if 'Source_id_1' in cat.colnames:
    for colname in cat.colnames:
      if colname.endswith('_1'):
        cat[colname].name = colname.replace('_1', '')
  
  # load images
  residual_fname = config.get('input', 'residual_image')
  clean_fname = config.get('input', 'clean_image')
  dirty_psf_fname = config.get('input', 'psf_image')
  residual_image = fits.getdata(residual_fname)[0,0]
  clean_image = fits.getdata(clean_fname)[0,0]
  dirty_psf_image = fits.getdata(dirty_psf_fname)[0,0]
  
  # set up wcs
  w_fourd = wcs.WCS(config.get('input', 'clean_image'))
  w_twod = w_fourd.dropaxis(3).dropaxis(2)
  header_twod = w_twod.to_header()
  image_size = clean_image.shape[0]
  pix_scale = np.abs(header_twod['CDELT1'])*galsim.degrees
  
  # setup galsim images
  dirty_psf_image_gs = galsim.ImageF(image_size, image_size, scale=pix_scale)
  dirty_psf_image_gs.wcs, origin = galsim.wcs.readFromFitsHeader(header_twod)
  #dirty_psf_image_gs.wcs = galsim.wcs.UniformWCS()
  dirty_psf_image_gs += galsim.ImageF(dirty_psf_image)
  
  residual_image_gs = galsim.ImageF(image_size, image_size, scale=pix_scale)
  residual_image_gs.wcs, origin = galsim.wcs.readFromFitsHeader(header_twod)
  residual_image_gs += galsim.ImageF(residual_image)
  
  clean_image = galsim.ImageF(clean_image)

  # get the beam information
  clean_image_header = fits.getheader(clean_fname)
  bmaj = clean_image_header['BMAJ']*galsim.degrees
  bmin = clean_image_header['BMIN']*galsim.degrees
  bpa = clean_image_header['BPA']*galsim.degrees - 90*galsim.degrees
  
  if (config.get('survey', 'psf_mode')=='wsclean'):
    psf_q = (bmin/galsim.degrees)/(bmaj/galsim.degrees)
    psf_pa = bpa
    psf_fwhm = bmaj
    psf = galsim.Gaussian(fwhm=psf_fwhm/galsim.arcsec)
    psf = psf.shear(q=psf_q, beta=psf_pa)
  elif (config.get('survey', 'psf_mode')=='hsm'):
    # create a gaussian hsm beam
    moms = galsim.hsm.FindAdaptiveMom(galsim.Image(dirty_psf_image, scale=pix_scale/galsim.arcsec))
    psf_sigma = moms.moments_sigma
    psf_e1 = moms.observed_shape.e1
    psf_e2 = moms.observed_shape.e2
    psf_e = np.sqrt(psf_e1**2. + psf_e2**2.)
    psf_q = (1. - psf_e)/(1. + psf_e)
    psf_pa = 0.5*np.arctan2(psf_e2, psf_e1)*galsim.radians
    psf = galsim.Gaussian(moms.moments_sigma)
    psf = psf.shear(moms.observed_shape)
    psf_fwhm = (pix_scale/galsim.arcsec)*psf.calculateFWHM()*galsim.arcsec
    psf = galsim.Gaussian(fwhm=psf_fwhm/galsim.arcsec)
    #psf = psf.shear(moms.observed_shape)
    psf = psf.shear(q=psf_q, beta=psf_pa)
  elif (config.get('survey', 'psf_mode')=='dirty'):
    # create a shapelet model of the dirty beam
    psf = galsim.FitShapelet(1, 6, dirty_psf_image_gs, dirty_psf_image_gs.center())
  
  # set up output columns
  g_1meas = np.empty([len(cat), n_shears, n_ellipticities])
  g_2meas = np.empty([len(cat), n_shears, n_ellipticities])
  
  e1_out_arr = np.array([])
  e2_out_arr = np.array([])
  e1_in_arr = np.array([])
  e2_in_arr = np.array([])
  
  idx=0

  for source_i, source in enumerate(cat):
  
    if not source_in_pointing(source, w_twod, clean_image.array.shape):
      print('Source {0}/{1} not in pointing {2}. Skipping.'.format(source_i, len(cat), config.get('input', 'pointing_name')))
      continue

    t_sourcestart = time.time()
    
    calibration_cat_fname = config.get('output', 'output_cat_dir')+'/{0}_supercals.fits'.format(source['Source_id'])
    calibration_output_cat = Table(names=calibration_output_columns, dtype=['S27']+(len(calibration_output_columns)-1)*[float])

    print('######################################')
    print('Source {0}/{1}:'.format(source_i, len(cat)))
    print('RA: {0}, DEC: {1}'.format(source['RA'], source['DEC']))
    print('Flux: '+('%.3e' % source['Total_flux'])+' Jy')
    print('Size: {0} arcsec'.format(source['Maj']*galsim.degrees/galsim.arcsec))
    print('######################################')
    print('e1_in\te1_out\t||\te2_in\te2_out')
    print('----------------||--------------------')
    for g_i, mod_g in enumerate(shears):
      print('Source: {0}/{1}, g: '.format(source_i, len(cat))+('%.2f' % mod_g))
      print('----------------||--------------------')
      for e_i, mod_e in enumerate(ellipticities):
        print('Source: {0}/{1}, g: '.format(source_i, len(cat))+('%.2f' % mod_g)+', e: '+('%.2f' % mod_e))
        print('----------------||--------------------')
        for o_i, theta in enumerate(orientations):
          
          if (mod_e == 0.0) and (o_i > 0):
            continue
          
          gal_gauss = galsim.Gaussian(fwhm=source['Maj']*galsim.degrees/galsim.arcsec, flux=source['Total_flux'])
          hlr = gal_gauss.getHalfLightRadius()
          gal = galsim.Exponential(half_light_radius=hlr, flux=source['Total_flux'], gsparams=big_fft_params)
          
          e1 = mod_e*np.cos(2.*theta)
          e2 = mod_e*np.sin(2.*theta)
          
          g1 = mod_g*np.cos(2.*shear_theta)
          g2 = mod_g*np.sin(2.*shear_theta)
                 
          ellipticity = galsim.Shear(e1=e1, e2=e2)
          shear = galsim.Shear(g1=g1, g2=g2)
          total_shear = ellipticity + shear
          gal = gal.shear(total_shear)
          obsgal = galsim.Convolve([gal, psf])
          
          x, y = w_twod.wcs_world2pix(source['RA'], source['DEC'], 0,)
          x = float(x)
          y = float(y)
          
          # Account for the fractional part of the position:
          ix = int(np.floor(x+0.5))
          iy = int(np.floor(y+0.5))
          offset = galsim.PositionD(x-ix, y-iy)
          
          # Create the sub-image for this galaxy
          stamp_size = get_stamp_size(source, pix_scale)
          options['stamp_size'] = stamp_size
          options['sersics_x0_start'] = stamp_size/2
          options['sersics_y0_start'] = stamp_size/2
          options['sersics_x0_max'] = stamp_size
          options['sersics_y0_max'] = stamp_size
          model_stamp = gal.drawImage(nx=stamp_size, ny=stamp_size, scale=pix_scale/galsim.arcsec, offset=offset)
          obsgal_stamp = obsgal.drawImage(nx=stamp_size, ny=stamp_size, scale=pix_scale/galsim.arcsec, offset=offset)
                  
          dirty_psf_stamp = dirty_psf_image[dirty_psf_image.shape[0]/2 - stamp_size/2:dirty_psf_image.shape[0]/2 + stamp_size/2, dirty_psf_image.shape[1]/2 - stamp_size/2:dirty_psf_image.shape[1]/2 + stamp_size/2]

          psf_image = galsim.Image(stamp_size, stamp_size, scale=pix_scale/galsim.arcsec)
          psf_stamp = psf.drawImage(image=psf_image)
          
          obsgal_stamp.setCenter(ix, iy)
          psf_stamp.setCenter(ix, iy)
          
          # Add the flux from the residual image to the sub-image
          bounds = obsgal_stamp.bounds & residual_image_gs.bounds
          
          # peak correction
          source_peak = (clean_image[bounds].array - residual_image_gs[bounds].array).max()
          flux_correction = source_peak/obsgal_stamp.array.max()
          
          obsgal_stamp = obsgal_stamp*flux_correction
          
          image_to_measure = obsgal_stamp[bounds] + residual_image_gs[bounds]
          
          if config.get('ring', 'doplots') and g_i==0:
            make_source_plot(config, bounds, clean_image, residual_image_gs, model_stamp, obsgal_stamp, image_to_measure, psf_stamp, dirty_psf_stamp, source, source_i, mod_e, theta)
                      
          weight = np.ones_like(obsgal_stamp.array) # ToDo: Should be from RMS map
          # Measure the shear with im3shape
          if config.get('input', 'measurement_method')=='im3shape':
            result, best_fit = analyze(image_to_measure.array, psf_stamp.array, options, weight=weight, ID=idx)
            result = process_im3shape_result(config, source, result, best_fit) 
          elif config.get('input', 'measurement_method')=='hsm':
            result = galsim.hsm.EstimateShear(image_to_measure.array, psf_stamp.array)
            result = {'g1_obs' : result.observed_shape.g1,
                      'g2_obs' : result.observed_shape.g2,
                      'e1_obs' : result.observed_shape.e1,
                      'e2_obs' : result.observed_shape.e2,
                      'radius' : result.moments_sigma,
                      'snr' : np.nan,
                      'likelihood' : np.nan,
                      'disc_A' : np.nan,
                      'disc_flux' : np.nan}
          
          calibration_output_cat.add_row([[source['Source_id']], [mod_g], [shear_theta], [mod_e], [theta], [g1], [g2], [e1], [e2], [result['g1_obs']], [result['g2_obs']], [result['e1_obs']], [result['e2_obs']], [result['radius']], [result['snr']], [result['likelihood']], [result['disc_A']], [result['disc_flux']]])
          
          print(('%.3f' % e1)+'\t'+('%.3f' % result['e1_obs'])+'\t||\t'+('%.3f' % e2)+'\t'+('%.3f' % result['e2_obs']))

          if g_i == 0:
            # also measure the actual shape in the clean image
            if config.get('input', 'measurement_method')=='im3shape':
              result, best_fit = analyze(clean_image[bounds].array, psf_stamp.array, options, weight=weight, ID=idx)
              result = process_im3shape_result(config, source, result, best_fit) 
            elif config.get('input', 'measurement_method')=='hsm':
              result = galsim.hsm.EstimateShear(image_to_measure.array, psf_stamp.array)
              result = {'g1_obs' : result.observed_shape.g1,
                        'g2_obs' : result.observed_shape.g2,
                        'e1_obs' : result.observed_shape.e1,
                        'e2_obs' : result.observed_shape.e2,
                        'radius' : result.moments_sigma,
                        'snr' : np.nan,
                        'likelihood' : np.nan,
                        'disc_A' : np.nan,
                        'disc_flux' : np.nan}
            image_output_cat.add_row([[source['Source_id']], [result['g1_obs']], [result['g2_obs']], [result['e1_obs']], [result['e2_obs']], [result['radius']], [result['snr']], [result['likelihood']], [result['disc_A']], [result['disc_flux']]])
          
          idx += 1
        print('----------------||--------------------')
  
    
    calibration_output_cat.write(calibration_cat_fname, format='fits', overwrite=True)
    
    t_sourceend = time.time()

    fits.setval(calibration_cat_fname, 'SOURCE_ID', value = source['Source_id'])
    fits.setval(calibration_cat_fname, 'RA', value = source['RA'])
    fits.setval(calibration_cat_fname, 'DEC', value = source['DEC'])
    fits.setval(calibration_cat_fname, 'TOTAL_FLUX', value = source['Total_flux'])
    fits.setval(calibration_cat_fname, 'MAJ', value = source['Maj'])
    fits.setval(calibration_cat_fname, 'MIN', value = source['Min'])
    fits.setval(calibration_cat_fname, 'PA', value = source['PA'])
    fits.setval(calibration_cat_fname, 'BMAJ', value = bmaj / galsim.degrees)
    fits.setval(calibration_cat_fname, 'BMIN', value = bmin / galsim.degrees)
    fits.setval(calibration_cat_fname, 'BPA', value = bpa / galsim.degrees)
    fits.setval(calibration_cat_fname, 'TIME_TAKEN', value = t_sourceend - t_sourcestart)
    
    print('Source {0} finished in '.format(source_i)+('%.2f seconds.' % (t_sourceend - t_sourcestart)))
    print('With an average of '.format(source_i)+('%.2f s' % (n_evals/float(t_sourceend - t_sourcestart)))+'/ring point.')
    print('--------------------------------------')
  
  image_output_cat.write(image_output_cat_fname, format='fits', overwrite=True)
  
if __name__ == '__main__':
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])
  
  runSuperCal(config)
