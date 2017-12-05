import pdb
import ConfigParser
import numpy as np
import sys
import os
import time
import cPickle as pickle
from math import ceil

from astropy.io import fits
from astropy.table import Table, join
import astropy.table as tb
from astropy import wcs

import galsim

sys.path.append('../simuclass')

from skymodel.skymodel_tools import setup_wcs

from matplotlib import pyplot as plt
plt.close('all')

image_output_columns = ('Source_id', 'g1_obs', 'g2_obs', 'e1_obs', 'e2_obs', 'radius', 'snr', 'likelihood', 'disc_A', 'disc_flux')  
calibration_output_columns = names=('Source_id', 'mod_g', 'theta_g', 'mod_e', 'theta_e', 'g1_inp', 'g2_inp', 'e1_inp', 'e2_inp', 'g1_obs', 'g2_obs', 'e1_obs', 'e2_obs', 'radius', 'snr', 'likelihood', 'disc_A', 'disc_flux')

big_fft_params = galsim.GSParams(maximum_fft_size=81488)

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

def processResult(config, source, result, best_fit):
  
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


def runSuperCal(config):
  print('######################################')
  print('       START POINTING {0}       '.format(config.get('input', 'pointing_name')))
  print('######################################')
  im3dir = config.get('im3shape', 'install_directory')

  sys.path.append(im3dir)

  from py3shape.analyze import analyze, count_varied_params
  from py3shape.options import Options
  
  options = Options(config.get('im3shape', 'ini_file'))

  n_shears = config.getint('ring','n_shears') # 1
  n_orientations = config.getint('ring','n_orientations') # 8
  n_ellipticities = config.getint('ring','n_ellipticities') # 15
  
  n_evals = n_shears*n_orientations*n_ellipticities
  
  # load catalogue positions
  cat = Table.read(config.get('input', 'catalogue'), format='fits')
  
  # ToDo: make this unnecessary.
  if 'Source_id_1' in cat.colnames:
    for colname in cat.colnames:
      if colname.endswith('_1'):
        cat[colname].name = colname.replace('_1', '')
  
  cat_snr = cat['Peak_flux']/cat['Resid_Isl_rms']
  cat = cat[cat_snr > config.getfloat('input', 'snr_cut')]
  
  # load residual image
  residual_fname = config.get('input', 'residual_image')
  clean_fname = config.get('input', 'clean_image')
  dirty_psf_fname = config.get('input', 'psf_image')
  residual_image = fits.getdata(residual_fname)[0,0]
  clean_image = fits.getdata(clean_fname)[0,0]
  dirty_psf_image = fits.getdata(dirty_psf_fname)[0,0]
  
  # set up wcs
  #w_twod = setup_wcs(config, ndim=2)
  w_fourd = wcs.WCS(config.get('input', 'clean_image'))
  w_twod = w_fourd.dropaxis(3).dropaxis(2)
  header_twod = w_twod.to_header()
  
  pix_scale = np.abs(header_twod['CDELT1'])*galsim.degrees
  image_size = clean_image.shape[0]
  
  residual_image_gs = galsim.ImageF(image_size, image_size, scale=pix_scale)
  im_center = residual_image_gs.bounds.trueCenter()

  # get the beam information
  clean_image_header = fits.getheader(clean_fname)
  bmaj = clean_image_header['BMAJ']*galsim.degrees
  bmin = clean_image_header['BMIN']*galsim.degrees
  bpa = clean_image_header['BPA']*galsim.degrees - 90*galsim.degrees
  
  if (config.get('survey', 'psf_mode')=='wsclean'):
    clean_psf_q = (bmin/galsim.degrees)/(bmaj/galsim.degrees)
    clean_psf_pa = bpa
    clean_psf_fwhm = bmaj

    clean_psf = galsim.Gaussian(fwhm=clean_psf_fwhm/galsim.arcsec)
    clean_psf = clean_psf.shear(q=clean_psf_q, beta=clean_psf_pa)
  elif (config.get('survey', 'psf_mode')=='hsm'):
    # create a gaussian hsm beam
    moms = galsim.hsm.FindAdaptiveMom(galsim.Image(dirty_psf_image, scale=pix_scale/galsim.arcsec))
    hsm_psf_sigma = moms.moments_sigma
    hsm_psf_e1 = moms.observed_shape.e1
    hsm_psf_e2 = moms.observed_shape.e2
    hsm_psf_e = np.sqrt(hsm_psf_e1**2. + hsm_psf_e2**2.)
    hsm_psf_q = (1. - hsm_psf_e)/(1. + hsm_psf_e)
    hsm_psf_pa = 0.5*np.arctan2(hsm_psf_e2, hsm_psf_e1)*galsim.radians

    hsm_psf = galsim.Gaussian(moms.moments_sigma)
    hsm_psf = hsm_psf.shear(moms.observed_shape)
    hsm_psf_fwhm = (pix_scale/galsim.arcsec)*hsm_psf.calculateFWHM()*galsim.arcsec
    hsm_psf = galsim.Gaussian(fwhm=hsm_psf_fwhm/galsim.arcsec)
    #hsm_psf = hsm_psf.shear(moms.observed_shape)
    hsm_psf = hsm_psf.shear(q=hsm_psf_q, beta=hsm_psf_pa)
    clean_psf = hsm_psf

  # Create a WCS for the galsim image
  residual_image_gs.wcs, origin = galsim.wcs.readFromFitsHeader(header_twod)

  residual_image = galsim.Image(residual_image)
  clean_image = galsim.Image(clean_image)

  residual_image_gs += residual_image

  orientations = np.linspace(0.e0, np.pi, n_orientations+1)
  orientations = orientations[:-1]
  ellipticities = np.linspace(0.e0, 0.7, n_ellipticities)
  ellipticities = ellipticities[::-1]
  shears = np.linspace(0.e0, 0.7, n_shears)
  shear_theta = 0.e0
  
  g_1meas = np.empty([len(cat), n_shears, n_ellipticities])
  g_2meas = np.empty([len(cat), n_shears, n_ellipticities])
  
  e1_out_arr = np.array([])
  e2_out_arr = np.array([])
  e1_in_arr = np.array([])
  e2_in_arr = np.array([])
  
  idx=0
  image_output_cat = Table(names=image_output_columns)
  image_output_cat_fname = config.get('output', 'output_cat_dir')+'/uncalibrated-shape-catalogue.fits'
  
  if not os.path.exists(config.get('output', 'output_cat_dir')):
    os.makedirs(config.get('output', 'output_cat_dir'))

  for source_i, source in enumerate(cat):
  
    if not source_in_pointing(source, w_twod, clean_image.array.shape):
      print('Source {0}/{1} not in pointing {2}. Skipping.'.format(source_i, len(cat), config.get('input', 'pointing_name')))
      continue
    
    '''
    if not source_i==4:
      print('!!!ONLY RUNNING DEBUG SOURCE!!!')
      continue
    '''
    t_sourcestart = time.time()
    
    calibration_cat_fname = config.get('output', 'output_cat_dir')+'/{0}-supercal.fits'.format(source['Source_id'])
    
    calibration_output_cat = Table(names=calibration_output_columns)
    #output_cat = Table(names=('Source_id', 'mod_g', 'theta_g', 'mod_e', 'theta_e', 'g1_inp', 'g2_inp', 'e1_inp', 'e2_inp'))
    #output_cat['theta_g'].unit = 'rad'
    #output_cat['theta_e'].unit = 'rad'
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
          #gal = galsim.Exponential(half_light_radius=hlr, flux=source['Total_flux'])
          gal = galsim.Sersic(n=4, half_light_radius=hlr, flux=source['Total_flux'], gsparams=big_fft_params)
          e1 = mod_e*np.cos(2.*theta)
          e2 = mod_e*np.sin(2.*theta)
          
          g1 = mod_g*np.cos(2.*shear_theta)
          g2 = mod_g*np.sin(2.*shear_theta)
          
          #output_row = Table([[source['Source_id']], [mod_g], [shear_theta], [mod_e], [theta], [g1], [g2], [e1], [e2]], names=('Source_id', 'mod_g', 'theta_g', 'mod_e', 'theta_e', 'g1_inp', 'g2_inp', 'e1_inp', 'e2_inp'))
          #if len(output_cat)==0:
          #  output_cat = output_row
          #else:
          #  output_cat = tb.join(output_cat, output_row)          
          ellipticity = galsim.Shear(e1=e1, e2=e2)
          shear = galsim.Shear(g1=g1, g2=g2)
          total_shear = ellipticity + shear
          
          gal = gal.shear(total_shear)
          '''
          psf = galsim.Gaussian(fwhm=bmaj/galsim.arcsec) # *PROBABLY* the clean beam PSF?
          psf_ellipticity = galsim.Shear(q=(bmin/galsim.arcsec)/(bmaj/galsim.arcsec), beta=bpa)
          psf = psf.shear(psf_ellipticity)
          '''
          obsgal = galsim.Convolve([gal, clean_psf])
          
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
          stamp = obsgal.drawImage(nx=stamp_size, ny=stamp_size, scale=pix_scale/galsim.arcsec, offset=offset)
                  
          dirty_psf_stamp = dirty_psf_image[dirty_psf_image.shape[0]/2 - stamp_size/2:dirty_psf_image.shape[0]/2 + stamp_size/2, dirty_psf_image.shape[1]/2 - stamp_size/2:dirty_psf_image.shape[1]/2 + stamp_size/2]

          clean_psf_image = galsim.Image(stamp_size, stamp_size, scale=pix_scale/galsim.arcsec)
          clean_psf_stamp = clean_psf.drawImage(image=clean_psf_image)
          '''
          clean_psf_stamp = galsim.Image(dirty_psf_stamp, scale=pix_scale/galsim.arcsec)
          '''
          clean_psf_stamp_array = clean_psf_stamp.array
          clean_psf_stamp_array = clean_psf_stamp_array/clean_psf_stamp_array.max()
          
          #psf_stamp = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=pix_scale/galsim.arcsec, offset=offset)
          
          stamp.setCenter(ix, iy)
          clean_psf_stamp.setCenter(ix, iy)
          #flux_correction = source['Peak_flux']/stamp.array.max()
          #stamp = stamp*flux_correction
          
          # Add the flux from the residual image to the sub-image
          bounds = stamp.bounds & residual_image_gs.bounds
          
          # integrated correction
          #flux_in_source = np.sum((clean_image[bounds].array)-(residual_image_gs[bounds].array))
          #flux_correction = flux_in_source/np.sum(stamp.array)
          
          # peak correction
          source_peak = clean_image[bounds].array.max() - residual_image_gs[bounds].array.max()
          flux_correction = source_peak/stamp.array.max()
          
          stamp = stamp*flux_correction
          
          image_to_measure = stamp[bounds] + residual_image_gs[bounds]
          
          if config.get('ring', 'doplots') and g_i==0:
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
            plt.imshow(dirty_psf_stamp, cmap='gnuplot2', interpolation='nearest', origin='lower')
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
            dirty_psf_oned = dirty_psf_stamp[dirty_psf_stamp.shape[0]/2,:]
            plt.plot(dirty_psf_oned*(source_peak/dirty_psf_oned.max()))
            f = plt.gca()
            f.axes.yaxis.set_ticklabels([])
            f.axes.xaxis.set_ticklabels([])
            #plt.text(10,10,str(np.sum(dirty_psf_stamp)), color='white')
            plt.xlim([0,166])
            
            fig.subplots_adjust(hspace=0, wspace=0)
            plt.savefig(config.get('output', 'output_cat_dir')+'/source_{0}_mode_{1}_rot_{2}.png'.format(source_i, mod_e, theta), dpi=300, bbox_inches='tight')
                      
          weight = np.ones_like(stamp.array) # ToDo: Should be from RMS map
          # Measure the shear with im3shape
          result, best_fit = analyze(image_to_measure.array, clean_psf_stamp.array, options, weight=weight, ID=idx)
          result = processResult(config, source, result, best_fit)
          calibration_output_cat.add_row([[source['Source_id']], [mod_g], [shear_theta], [mod_e], [theta], [g1], [g2], [e1], [e2], [result['g1_obs']], [result['g2_obs']], [result['e1_obs']], [result['e2_obs']], [result['radius']], [result['snr']], [result['likelihood']], [result['disc_A']], [result['disc_flux']]])
          #join(output_cat, results_row)

          if g_i == 0:
            # also measure the actual shape in the clean image
            result, best_fit = analyze(clean_image[bounds].array, clean_psf_stamp.array, options, weight=weight, ID=idx)
            result = processResult(config, source, result, best_fit) 
            image_output_cat.add_row([[source['Source_id']], [result['g1_obs']], [result['g2_obs']], [result['e1_obs']], [result['e2_obs']], [result['radius']], [result['snr']], [result['likelihood']], [result['disc_A']], [result['disc_flux']]])
            #output_cat = tb.join(output_cat, results_row)
          
          print(('%.3f' % e1)+'\t'+('%.3f' % result['e1_obs'])+'\t||\t'+('%.3f' % e2)+'\t'+('%.3f' % result['e2_obs']))
          
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
