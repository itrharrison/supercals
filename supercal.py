import pdb
import ConfigParser
import numpy as np
import sys
import cPickle as pickle

from astropy.io import fits
from astropy.table import Table
from astropy import wcs

import galsim

sys.path.append('../simuCLASS')

from skymodel.skymodel_tools import setup_wcs

from matplotlib import pyplot as plt
plt.close('all')

def get_stamp_size(source, pixel_scale):

  stamp_size = 3.* (source['size'] / galsim.arcsec) / (pixel_scale / galsim.arcsec)

  return stamp_size

def runSuperCal(config):
  im3dir = config.get('im3shape', 'install_directory')

  sys.path.append(im3dir)

  from py3shape.analyze import analyze, count_varied_params
  from py3shape.options import Options
  
  options = Options(config.get('im3shape', 'ini_file'))

  n_shears = config.getint('ring','n_shears') # 1
  n_orientations = config.getint('ring','n_orientations') # 8
  n_ellipticities = config.getint('ring','n_ellipticities') # 15
  
  # load catalogue positions
  cat = Table.read(config.get('input', 'catalogue'), format='fits')
  if config.get('input', 'catalogue_origin')=='pybdsm':
    cat['ra_abs'] = cat['RA']
    cat['dec_abs']= cat['DEC']
    cat['size'] = cat['Maj'] * galsim.degrees
    cat['integrated_flux'] = cat['Total_flux']
  
  # load residual image
  residual_fname = config.get('input', 'residual_image')
  clean_fname = config.get('input', 'clean_image')
  residual_image = fits.getdata(residual_fname)[0,0]
  clean_image = fits.getdata(clean_fname)[0,0]

  # get the beam information
  clean_image_header = fits.getheader(clean_fname)
  bmaj = clean_image_header['BMAJ']*galsim.degrees
  bmin = clean_image_header['BMIN']*galsim.degrees
  bpa = clean_image_header['BPA']*galsim.degrees
  
  # set up wcs
  #w_twod = setup_wcs(config, ndim=2)
  w_fourd = wcs.WCS(config.get('input', 'clean_image'))
  w_twod = w_fourd.dropaxis(3).dropaxis(2)
  header_twod = w_twod.to_header()
  
  pixel_scale = abs(header_twod['CDELT1'])*galsim.degrees
  image_size = clean_image.shape[0]
  
  '''
  pixel_scale = config.getfloat('skymodel', 'pixel_scale')*galsim.arcsec
  fov = config.getfloat('skymodel', 'field_of_view')*galsim.arcmin
  image_size = int((fov/galsim.arcmin)/(pixel_scale/galsim.arcmin))
  '''
  
  full_image = galsim.ImageF(image_size, image_size, scale=pixel_scale)
  im_center = full_image.bounds.trueCenter()

  # Create a WCS for the galsim image
  full_image.wcs, origin = galsim.wcs.readFromFitsHeader(header_twod)

  
  residual_image = galsim.Image(residual_image)
  clean_image = galsim.Image(clean_image)

  full_image += residual_image

  orientations = np.linspace(0.e0, np.pi, n_orientations)
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

  for source_i, source in enumerate(cat):
    for g_i, mod_g in enumerate(shears):
      for e_i, mod_e in enumerate(ellipticities):
        for o_i, theta in enumerate(orientations-1):
          
          gal = galsim.Exponential(scale_radius=source['size']/galsim.arcsec, flux=source['integrated_flux'])
          
          e1 = mod_e*np.cos(2.*theta)
          e2 = mod_e*np.sin(2.*theta)
          
          g1 = mod_g*np.cos(2.*shear_theta)
          g2 = mod_g*np.sin(2.*shear_theta)
          
          ellipticity = galsim.Shear(e1=e1, e2=e2)
          shear = galsim.Shear(g1=g1, g2=g2)
          total_shear = ellipticity + shear
          
          gal = gal.shear(total_shear)
          
          psf = galsim.Gaussian(fwhm=bmaj/galsim.arcsec) # *PROBABLY* the clean beam PSF?
          psf_ellipticity = galsim.Shear(q=(bmin/galsim.arcsec)/(bmaj/galsim.arcsec), beta=bpa)
          psf = psf.shear(psf_ellipticity)
          
          obsgal = galsim.Convolve([gal, psf])
          
          x, y = w_twod.wcs_world2pix(source['ra_abs'], source['dec_abs'], 0,)
          x = float(x)
          y = float(y)
          
          
          # Account for the fractional part of the position:
          ix = int(np.floor(x+0.5))
          iy = int(np.floor(y+0.5))
          offset = galsim.PositionD(x-ix, y-iy)
          
          # Create the sub-image for this galaxy
          stamp_size = get_stamp_size(source, pixel_scale)
          stamp = obsgal.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale/galsim.arcsec, offset=offset)
          psf_stamp = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale/galsim.arcsec, offset=offset)
          
          stamp.setCenter(ix, iy)
          psf_stamp.setCenter(ix, iy)
          
          # Add the flux from the residual image to the sub-image
          bounds = stamp.bounds & full_image.bounds
          
          '''
          plt.figure(1)
          plt.subplot(141)
          plt.imshow(clean_image[bounds].array, cmap='afmhot', interpolation='nearest')
          plt.title('CLEAN')
          plt.axis('off')
          plt.subplot(142)
          plt.imshow(full_image[bounds].array, cmap='afmhot', interpolation='nearest')
          plt.title('Residual')
          plt.axis('off')
          plt.subplot(143)
          plt.imshow(stamp.array, cmap='afmhot', interpolation='nearest')
          plt.title('Model')
          plt.axis('off')
          plt.subplot(144)
          plt.imshow(stamp.array + full_image[bounds].array, cmap='afmhot', interpolation='nearest')
          plt.title('Model + Residual')
          plt.axis('off')
          plt.savefig('plots/rot_{0}.png'.format(o_i), dpi=160, bbox_inches='tight')
          '''
          
          stamp[bounds] += full_image[bounds]
          weight = np.ones_like(stamp.array) # ToDo: Should be from RMS map
          # Measure the shear with im3shape
          result, best_fit = analyze(stamp.array, psf_stamp.array, options, weight=weight, ID=idx)
          result = result.as_dict(0, count_varied_params(options))
          g_1meas[source_i, g_i, e_i] += result[0]['e1']
          g_2meas[source_i, g_i, e_i] += result[0]['e2']
          
          print(e1, result[0]['e1'])
          print(e2, result[0]['e2'])
          
          e1_out_arr = np.append(e1_out_arr,result[0]['e1'])
          e2_out_arr = np.append(e2_out_arr,result[0]['e2'])
          
          e1_in_arr = np.append(e1_in_arr, e1)
          e2_in_arr = np.append(e2_in_arr, e2)
          
          
          idx += 1

        g_1meas[source_i, g_i, e_i] = g_1meas[source_i, g_i, e_i]/n_orientations
        g_2meas[source_i, g_i, e_i] = g_2meas[source_i, g_i, e_i]/n_orientations

if __name__ == '__main__':
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])
  
  runSuperCal(config)
