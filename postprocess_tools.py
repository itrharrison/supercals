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

  '''
  source['e1_corrected'] = (source['e1_obs'] - c_e1)/(1.e0 + m_e1)
  source['e2_corrected'] = (source['e2_obs'] - c_e2)/(1.e0 + m_e2)
  
  # im3shape is a maximum likelihood code and doesn't give any output about the curvature...
  sigma2_e1 = 0.e0
  sigma2_e2 = 0.e0

  source['sigma2_e1_corrected'] = (sigma2_e1 + sigma2_c_e1 + sigma2_m_e1*((c_e1 - source['e1_obs'])/(1.e0 + m_e1))**2.)/(1.e0 + m_e1)**2.
  source['sigma2_e2_corrected'] = (sigma2_e2 + sigma2_c_e2 + sigma2_m_e2*((c_e2 - source['e2_obs'])/(1.e0 + m_e2))**2.)/(1.e0 + m_e2)**2.

  output_cat.add_row([config.get('input', 'pointing_name'), m_e1, c_e1, sigma2_m_e1, sigma2_c_e1,
                                                            m_e2, c_e2, sigma2_m_e2, sigma2_c_e2])
  '''

  calibration_fit = {
                      'm_e1' : m_e1,
                      'm_e2' : m_e2,
                      'c_e1' : c_e1,
                      'c_e2' : c_e2,
                      'sigma2_m_e1' : sigma2_m_e1,
                      'sigma2_m_e2' : sigma2_m_e2,
                      'sigma2_c_e1' : sigma2_c_e1,
                      'sigma2_c_e2' : sigma2_c_e2,
                      }

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


  f.savefig(config.get('output', 'output_plot_dir')+'/calib_surface_plots.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])
  
  make_m_and_c(config)
  make_calib_surface_plots(config)
