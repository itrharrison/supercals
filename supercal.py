import pdb
import ConfigParser
import numpy as np
import sys

from astropy.io import fits

import galsim

from simuclass.skymodel_tools import setup_wcs

def runSuperCal(config)
  im3dir = config.get('im3shape', 'install_directory')

  sys.path.append(im3dir)

  from py3shape.analyze import analyze, count_varied_params
  from py3shape.options import Options

  n_shears = 15
  n_orientations = 8

  # load catalogue positions

  # load residual image
  full_image = residual_image

  orientations = np.linspace(0.e0, np.pi, n_orientations)
  ellipticities = np.linspace(0.e0, 0.7, n_ellipticities)
  shears = [0]
  shear_theta = 0.e0

  for gal_i, gal in enumerate(cat):
    for g_i, mod_g in enumerate(shears):
      for e_i, mod_e in enumerate(ellipticities):
        for o_i, theta in enumerate(orientations):
          
          gal = galsim.Exponential(scale_radius=gal_r0[gal_i], flux=gal_flux[gal_i])
          
          e1 = mod_e*np.cos(2.*theta)
          e2 = mod_e*np.sin(2.*theta)
          
          g1 = mod_g*np.cos(2.*shear_theta)
          g2 = mod_g*np.sin(2.*shear_theta)
          
          ellipticity = galsim.Shear(e1=e1, e2=e2)
          shear = galsim.Shear(g1=g1, g2=g2)
          total_shear = ellipticity + shear
          
          gal = gal.shear(total_shear)
          
          psf = galsim.Gaussian(psf_fwhm) # *PROBABLY* the clean beam PSF?
          
          obsgal = galsim.Convolve([gal, psf])
          
          x, y = w_twod.wcs_world2pix(gal_ra[i], gal_dec[i], 0,)
          x = float(x)
          y = float(y)
          
          # Account for the fractional part of the position:
          ix = int(np.floor(x+0.5))
          iy = int(np.floor(y+0.5))
          ix_arr[i] = ix
          iy_arr[i] = iy
          offset = galsim.PositionD(x-ix, y-iy)
          
          # Create the sub-image for this galaxy
          stamp = obsgal.drawImage(scale=pixel_scale/galsim.arcsec, offset=offset)
          psf_stamp = psf.drawImage(scale=pixel_scale/galsim.arcsec, offset=offset)
          
          stamp.setCenter(ix, iy)
          psf_stamp.setCenter(ix, iy)
          
          # Add the flux from the residual image to the sub-image
          bounds = stamp.bounds & full_image.bounds
          stamp[bounds] += full_image[bounds]
          
          weight = np.ones_like(stamp.array) # ToDo: Should be from RMS map
          
          # Measure the shear with im3shape
          result, best_fit = analyze(stamp, psf_stamp, options, weight=weight, ID=idx)
          result = result.as_dict(1, count_varied_params(options))
				  g_1meas[gal_i, g_i, e_i] += result[0]['e1']
				  g_2meas[gal_i, g_i, e_i] += result[0]['e2']
				
			  g_1meas[gal_i, g_i, e_i] = g_1meas[gal_i, g_i, e_i]/n_orientations
			  g_2meas[gal_i, g_i, e_i] = g_2meas[gal_i, g_i, e_i]/n_orientations			

if __name__ == '__main__':
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])
  
  runSuperCal(config)
