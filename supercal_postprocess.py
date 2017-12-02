import pdb
try:
  import ConfigParser
except ImportError:
  import configparser as ConfigParser
import numpy as np
import sys
import time
import pickle

from scipy import interpolate
from scipy.optimize import curve_fit

from astropy.io import fits
from astropy.table import Table
from astropy import wcs

sys.path.append('../simuCLASS')

from matplotlib import pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1 import AxesGrid

from astropy import wcs

#rc('text', usetex=True)
#rc('font', family='serif')
#rc('font', size=11)

plt.close('all') # tidy up any unshown plots

def calculate_corrected_ellipticity(source, config):

  source_cat_fname = config.get('output', 'source_cat_dir')+'/{0}'.format(source['Source_id'])
  source_cat = Table.read(source_cat_fname, format='fits')
  


  

def make_m_and_c(source, config):
  
  def flin(x, m, c):
    return m*x + c

  '''
  - read wl gold catalogue
  - for a given source in wl-gold catalogue:
    - for all the pointings for that source:
      - calculate the m and c for that source in that pointing
      - calculate the corrected ellipticity
      - calculate the error on the corrected ellipticity
    - calculate the mean averaged (inverse variance weighted?) corrected ellipticity over all pointings
  
  '''

  pointing_cat_fname = config.get('output', 'output_cat_dir')+'/{0}-supercal.fits'.format(source['Source_id'])
  source_cat_fname = config.get('output', 'source_cat_dir')+'/{0}'.format(source['Source_id'])

  input_cat = Table.read(pointing_cat_fname, format='fits')
  output_cat = Table.read(source_cat_fname, format='fits')

  popt_e1, pcov_e1 = curve_fit(flin, pointing_cat['e1_inp'], pointing_cat['e1'] - pointing_cat['e1_inp'])
  popt_e2, pcov_e2 = curve_fit(flin, pointing_cat['e2_inp'], pointing_cat['e2'] - pointing_cat['e2_inp'])
  perr_e1 = np.sqrt(np.diag(pcov_e1))
  perr_e2 = np.sqrt(np.diag(pcov_e2))

  m_e1 = popt_e1[0]
  c_e1 = popt_e1[1]
  sigma2_m_e1 = perr_e1[0]**2.
  sigma2_c_e1 = perr_e1[1]**2.

  m_e2 = popt_e2[0]
  c_e2 = popt_e2[1]
  sigma2_m_e2 = perr_e2[0]**2.
  sigma2_c_e2 = perr_e2[1]**2.

  source['e1_corrected'] = (source['e1_obs'] - c_e1)/(1.e0 + m_e1)
  source['e2_corrected'] = (source['e2_obs'] - c_e2)/(1.e0 + m_e2)
  
  # im3shape is a maximum likelihood code and doesn't give any output about the curvature...
  sigma2_e1 = 0.e0
  sigma2_e2 = 0.e0

  source['sigma2_e1_corrected'] = (sigma2_e1 + sigma2_c_e1 + sigma2_m_e1*((c_e1 - source['e1_obs'])/(1.e0 + m_e1))**2.)/(1.e0 + m_e1)**2.
  source['sigma2_e2_corrected'] = (sigma2_e2 + sigma2_c_e2 + sigma2_m_e2*((c_e2 - source['e2_obs'])/(1.e0 + m_e2))**2.)/(1.e0 + m_e2)**2.

  output_cat.add_row([m_e1, c_e1, sigma2_m_e1, sigma2_c_e1,
                      m_e2, c_e2, sigma2_m_e2, sigma2_c_e2])
  '''
  cat = Table.read(config.get('input', 'catalogue'), format='fits')
  cat_snr = cat['Peak_flux']/cat['Resid_Isl_rms']
  cat = cat[cat_snr > config.getfloat('input', 'snr_cut')]
  cat_supercals_cols = Table(names=('supercals_m_e1','supercals_c_e1',
                                    'E_supercals_m_e1','E_supercals_c_e1',
                                    'supercals_m_e2','supercals_c_e2',
                                    'E_supercals_m_e2','E_supercals_c_e2'))

  if config.getboolean('output','do_ein_eout_plots'):
    clean_fname = config.get('input', 'clean_image')
    clean_image = fits.getdata(clean_fname)[0,0]
    w_fourd = wcs.WCS(config.get('input', 'clean_image'))
    w_twod = w_fourd.dropaxis(3).dropaxis(2)

  for source_i, source in enumerate(cat):
    print(source_i)
    supercals_cat_fname = config.get('output', 'output_cat_dir')+'/{0}_supercal_output.fits'.format(source['Source_id'])
    supercals_cat = Table.read(supercals_cat_fname, format='fits')

    popt_e1, pcov_e1 = curve_fit(flin, supercals_cat['e1_inp'], supercals_cat['e1'] - supercals_cat['e1_inp'])
    popt_e2, pcov_e2 = curve_fit(flin, supercals_cat['e2_inp'], supercals_cat['e2'] - supercals_cat['e2_inp'])
    perr_e1 = np.sqrt(np.diag(pcov_e1))
    perr_e2 = np.sqrt(np.diag(pcov_e2))
    cat_supercals_cols.add_row([popt_e1[0],popt_e1[1],
                                perr_e1[0],perr_e1[1],
                                popt_e2[0],popt_e2[1],
                                perr_e2[0],perr_e2[1]])
    if config.getboolean('output','do_ein_eout_plots'):
      ein_eout_plot_fname = config.get('output', 'output_plot_dir')+'/{0}_supercals_ein_eout.png'.format(int(supercals_cat['Source_id'][0]))
      make_ein_eout_plots(supercals_cat, popt_e1[0], popt_e1[1], popt_e2[0], popt_e2[1], ein_eout_plot_fname)

  if 'supercals_m_e1' in cat.colnames:
    cat['supercals_m_e1'] = cat_supercals_cols['supercals_m_e1']
    cat['supercals_c_e1'] = cat_supercals_cols['supercals_c_e1']
    cat['E_supercals_m_e1'] = cat_supercals_cols['E_supercals_m_e1']
    cat['E_supercals_c_e1'] = cat_supercals_cols['E_supercals_c_e1']
    cat['supercals_m_e2'] = cat_supercals_cols['supercals_m_e2']
    cat['supercals_c_e2'] = cat_supercals_cols['supercals_c_e2']
    cat['E_supercals_m_e2'] = cat_supercals_cols['E_supercals_m_e2']
    cat['E_supercals_c_e2'] = cat_supercals_cols['E_supercals_c_e2']
  else:
    cat.add_columns([cat_supercals_cols['supercals_m_e1'],
                     cat_supercals_cols['supercals_c_e1'],
                     cat_supercals_cols['E_supercals_m_e1'],
                     cat_supercals_cols['E_supercals_c_e1'],
                     cat_supercals_cols['supercals_m_e2'],
                     cat_supercals_cols['supercals_c_e2'],
                     cat_supercals_cols['E_supercals_m_e2'],
                     cat_supercals_cols['E_supercals_c_e2']])
  
  cat.write(config.get('input', 'catalogue'), format='fits', overwrite=True)

  '''

def make_ein_eout_plots(source, m_e1, c_e1, m_e2, c_e2, fname, stamp=None):

  plt.close('all')
  if stamp is None:
    plt.figure(1, figsize=(5, 3.75))
    plt.subplot(111)
  else:
    plt.figure(1, figsize=(10, 3.75))
    plt.subplot(121)
  plt.axhline(0, color='k', linestyle='--')
  plt.plot(source['e1_inp'], source['e1'] - source['e1_inp'], 'o', color='powderblue', label='$e_{1}$')
  plt.plot(source['e2_inp'], source['e2'] - source['e2_inp'], '+', color='lightcoral', label='$e_{2}$')
  x = np.linspace(-1,1,32)
  y_e1 = m_e1*x + c_e1
  y_e2 = m_e2*x + c_e2
  plt.plot(x, y_e1, '-', color='powderblue')
  plt.plot(x, y_e2, '-', color='lightcoral')
  plt.xlabel('$\mathrm{Input \, Ellipticity} \, \, e^{\\rm inp}_{i}$')
  plt.ylabel('$\mathrm{Ellipticity \, Bias} \, \, e^{\\rm obs}_{i} - e^{\\rm inp}_{i}$')
  plt.xlim([-1,1])
  plt.xticks([-1,-0.5,0,0.5,1], ('-1','-0.5','0','0.5','1'))
  plt.ylim([-1,1])
  plt.yticks([-1,-0.5,0,0.5,1], ('-1','-0.5','0','0.5','1'))
  plt.legend(frameon=False)

  if stamp is not None:
    plt.subplot(122)
    plt.imshow(stamp, cmap='afmhot')
    plt.axis('off')


  '''
  plt.subplot(122)
  plt.plot(source['e1_inp'], source['e1'] - source['e1_inp'], 'o', color='powderblue')
  x = np.linspace(-1,1,32)
  y = m*x + c
  plt.plot(x, y, 'k-')
  plt.axhline(0, color='k', linestyle='--')
  plt.xlabel('$\mathrm{Input \, Ellipticity} \, \, e^{\\rm inp}_{2}$')
  plt.xlim([-1,1])
  plt.ylim([-1,1])
  #plt.ylabel('$\mathrm{Ellipticity \, Bias} e^{\\rm obs}_{1} - e^{\\rm inp}_{1}$')
  '''

  plt.savefig(fname, dpi=300, bbox_inches='tight')

def make_calib_surface_plots(config):

  cat = Table.read(config.get('input', 'catalogue'), format='fits')

  plt.close('all')

  f, axarr = plt.subplots(2,2)
  p = axarr[0,0].scatter(cat['RA'], cat['DEC'], c=abs(cat['supercals_m_e1']), cmap='gnuplot2')
  axarr[0,0].set_title('$|m_{e_1}|$')
  f.colorbar(p, ax=axarr[0,0], pad=0)
  p = axarr[0,1].scatter(cat['RA'], cat['DEC'], c=abs(cat['supercals_m_e2']), cmap='gnuplot2')
  axarr[0,1].set_title('$|m_{e_2}|$')
  f.colorbar(p, ax=axarr[0,1], pad=0)
  p = axarr[1,0].scatter(cat['RA'], cat['DEC'], c=abs(cat['supercals_c_e1']), cmap='gnuplot2')
  axarr[1,0].set_title('$|c_{e_1}|$')
  f.colorbar(p, ax=axarr[1,0], pad=0)
  axarr[1,0].set_xlabel('$\mathrm{RA \, \, [deg]}$')
  axarr[1,0].set_ylabel('$\mathrm{Dec \, \, [deg]}$')
  p = axarr[1,1].scatter(cat['RA'], cat['DEC'], c=abs(cat['supercals_c_e2']), cmap='gnuplot2')
  axarr[1,1].set_title('$|c_{e_2}|$')
  f.colorbar(p, ax=axarr[1,1], pad=0)

  plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
  plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)


  f.savefig(config.get('output', 'output_plot_dir')+'/calib_plots.png', dpi=300, bbox_inches='tight')


  '''
  grid = AxesGrid(fig, 143,  # similar to subplot(143)
                  nrows_ncols=(2, 2),
                  axes_pad=0.1,
                  label_mode="1",
                  share_all=True,
                  cbar_location="bottom",
                  cbar_mode="each",
                  cbar_size="7%",
                  cbar_pad="2%",
                  )

  p = grid[0].scatter(cat['RA'], cat['DEC'], c=cat['supercals_m_e1'])
  grid.cbar_axes[0].colorbar(p)
  p = grid[1].scatter(cat['RA'], cat['DEC'], c=cat['supercals_m_e2'])
  grid.cbar_axes[1].colorbar(p)
  p = grid[2].scatter(cat['RA'], cat['DEC'], c=cat['supercals_c_e1'])
  grid.cbar_axes[2].colorbar(p)
  p = grid[3].scatter(cat['RA'], cat['DEC'], c=cat['supercals_c_e2'])
  grid.cbar_axes[3].colorbar(p)

  plt.savefig('calib_plots.png', dpi=300, bbox_inches='tight')
  '''

  '''
  finterp_m = interpolate.interp2d(cat['RA'], cat['DEC'], cat['supercals_m_e1'], kind='cubic')
  finterp_c = interpolate.interp2d(cat['RA'], cat['DEC'], cat['supercals_c_e1'], kind='cubic')

  ran = np.linspace(ra_min, ra_max, 512)
  decn = np.linspace(dec_min, dec_max, 512)

  ra_new, dec_new = np.meshgrid(ran, decn)
  m_surf = finterp_m(ran, decn)
  c_surf = finterp_c(ran, decn)

  plt.subplot(121)
  plt.scatter(ra_new, dec_new, c=m_surf)

  plt.subplot(122)
  plt.scatter(ra_new, dec_new, c=c_surf)
  '''

'''
def get_stamp_size(source, pixel_scale):

  stamp_size = 10.* (source['Maj']*galsim.degrees / galsim.arcsec) / (pixel_scale / galsim.arcsec)

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
    t_sourcestart = time.time()
    output_cat = Table(names=('mod_g', 'theta_g', 'mod_e', 'theta_e', 'g1_inp', 'g2_inp', 'e1_inp', 'e2_inp', 'e1', 'e2', 'radius', 'snr', 'disc_A'))
    output_cat['theta_g'].unit = 'rad'
    output_cat['theta_e'].unit = 'rad'
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
          
          gal = galsim.Exponential(scale_radius=source['Maj']*galsim.degrees/galsim.arcsec, flux=source['Total_flux'])
          
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
          
          x, y = w_twod.wcs_world2pix(source['RA'], source['DEC'], 0,)
          x = float(x)
          y = float(y)
          
          
          # Account for the fractional part of the position:
          ix = int(np.floor(x+0.5))
          iy = int(np.floor(y+0.5))
          offset = galsim.PositionD(x-ix, y-iy)
          
          # Create the sub-image for this galaxy
          stamp_size = get_stamp_size(source, pixel_scale)
          options['stamp_size'] = stamp_size
          options['sersics_x0_start'] = stamp_size/2
          options['sersics_y0_start'] = stamp_size/2
          options['sersics_x0_max'] = stamp_size
          options['sersics_y0_max'] = stamp_size
          stamp = obsgal.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale/galsim.arcsec, offset=offset)
          psf_stamp = psf.drawImage(nx=stamp_size, ny=stamp_size, scale=pixel_scale/galsim.arcsec, offset=offset)
          
          stamp.setCenter(ix, iy)
          psf_stamp.setCenter(ix, iy)
          
          flux_correction = source['Peak_flux']/stamp.array.max()
          stamp = stamp*flux_correction
          
          # Add the flux from the residual image to the sub-image
          bounds = stamp.bounds & full_image.bounds
          
          if config.get('ring', 'doplots') and g_i==1:
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
            plt.savefig('plots/source_{0}.png'.format(source_i), dpi=160, bbox_inches='tight')
          
          stamp[bounds] += full_image[bounds]
          weight = np.ones_like(stamp.array) # ToDo: Should be from RMS map
          # Measure the shear with im3shape
          result, best_fit = analyze(stamp.array, psf_stamp.array, options, weight=weight, ID=idx)
          result = result.as_dict(0, count_varied_params(options))
          e1_obs = result[0]['e1']
          e2_obs = result[0]['e2']
          radius = result[0]['radius']
          snr = result[0]['snr']
          disc_A = result[0]['disc_A']
          
          output_cat.add_row([mod_g, shear_theta, mod_e, theta, g1, g2, e1, e2, e1_obs, e2_obs, radius, snr, disc_A])
          
          print(('%.3f' % e1)+'\t'+('%.3f' % e1_obs)+'\t||\t'+('%.3f' % e2)+'\t'+('%.3f' % e2_obs))
          
          idx += 1
        print('----------------||--------------------')
  
    cat_fname = '{0}_supercals_output.fits'.format(source['Source_id'])
    output_cat.write(cat_fname, format='fits', overwrite=True)
    
    t_sourceend = time.time()

    fits.setval(cat_fname, 'SOURCE_ID', value = source['Source_id'])
    fits.setval(cat_fname, 'RA', value = source['RA'])
    fits.setval(cat_fname, 'DEC', value = source['DEC'])
    fits.setval(cat_fname, 'TOTAL_FLUX', value = source['Total_flux'])
    fits.setval(cat_fname, 'MAJ', value = source['Maj'])
    fits.setval(cat_fname, 'MIN', value = source['Min'])
    fits.setval(cat_fname, 'PA', value = source['PA'])
    fits.setval(cat_fname, 'BMAJ', value = bmaj / galsim.degrees)
    fits.setval(cat_fname, 'BMIN', value = bmin / galsim.degrees)
    fits.setval(cat_fname, 'BPA', value = bpa / galsim.degrees)
    fits.setval(cat_fname, 'TIME_TAKEN', value = t_sourceend - t_sourcestart)
    
    print('Source {0} finished in '.format(source_i)+('%.2f seconds.' % (t_sourceend - t_sourcestart)))
    print('--------------------------------------')
'''
if __name__ == '__main__':
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])
  
  make_m_and_c(config)
  make_calib_surface_plots(config)
