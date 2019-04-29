import numpy as np
from scipy import interpolate
import configparser as ConfigParser
import sys
import os
import pdb
from scipy.optimize import curve_fit

from matplotlib import pyplot as plt
from matplotlib import rc

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

def bias_surface_plots(zobs, zin, xin, yin, z_bias_surface, obs_point, plot_fname):

    delta_z = zobs - zin

    xnew = np.linspace(xin.min(), xin.max(), 128)
    ynew = np.linspace(yin.min(), yin.max(), 128)

    xxnew, yynew = np.meshgrid(xnew, ynew)

    z_bias = z_bias_surface.ev(xxnew, yynew)

    plt.close('all')
    plt.figure(1, figsize=(4.5, 3.75))
    plt.scatter(xxnew,yynew, c=z_bias, edgecolor='none')
    cb = plt.colorbar()
    cb.set_label('Data')
    plt.scatter(xin, yin, c=delta_z, edgecolors='k')
    cb = plt.colorbar()
    cb.set_label('Fit')
    plt.scatter(obs_point[0], obs_point[1], c='lightcoral', edgecolors='k')
    plt.xlabel('$e_{1}$')
    plt.ylabel('$e_{2}$')
    plt.xlim([xin.min(),xin.max()])
    plt.ylim([yin.min(),yin.max()])
    plt.savefig(plot_fname, dpi=300, bbox_inches='tight')

def bias_surface_1d(xin, xout):

  def flin(x, m, c):
    return m*x + c

  popt_e1, pcov_e1 = curve_fit(flin, xin, xout - xin)
  perr_e1 = np.sqrt(np.diag(pcov_e1))

  m_e1 = popt_e1[0]
  c_e1 = popt_e1[1]

  return m_e1, c_e1

def bias_surface_2d(zobs, zin, xin, yin, kx=2, ky=2):

  delta_z = zobs - zin

  z_bias_surface = interpolate.SmoothBivariateSpline(xin, yin, delta_z, kx=kx, ky=ky)

  return z_bias_surface

def make_cat_calibration_plots(cat, base_dir='./', name=''):

  if not os.path.exists(base_dir+'/plots/'):
    os.mkdir(base_dir+'/plots/')

  beam_a = 1.9
  beam_b = 1.5
  beam_mode = (beam_a - beam_b)/(beam_a + beam_b)
  beam_pa = 1.39626 - np.pi/2.
  beam_e1 = beam_mode*np.cos(2.*beam_pa)
  beam_e2 = beam_mode*np.sin(2.*beam_pa)

  plt.close('all')

  plt.figure(1, figsize=(4.5, 3.75))
  plt.plot(cat['e1_uncalibrated_supercals'],cat['e2_uncalibrated_supercals'],'o', label='Uncalilbrated', ms=3)
  plt.plot(cat['e1_calibrated_supercals'],cat['e2_calibrated_supercals'],'o', label='Calibrated', ms=3)
  plt.plot(cat['e1_uncalibrated_supercals'].mean(), cat['e2_uncalibrated_supercals'].mean(), 'o', mfc='blue', mec='k', ms=10)
  plt.plot(cat['e1_calibrated_supercals'].mean(), cat['e2_calibrated_supercals'].mean(), 'o', mfc='orange', mec='k', ms=10)
  plt.plot(beam_e1, beam_e2, 'k+', label='Beam')
  unitcirc = plt.Circle((0,0),1.0, facecolor='none', edgecolor='k', linestyle='dashed', zorder=99)#, alpha=0.4)
  ax = plt.gca()
  ax.add_artist(unitcirc)
  plt.axvline(0, linestyle='dashed', color='k')#, alpha=0.4)
  plt.axhline(0, linestyle='dashed', color='k')#, alpha=0.4)
  plt.legend(loc='lower left')
  plt.xlabel('$e_1$')
  plt.ylabel('$e_2$')
  plt.xlim([-1,1])
  plt.ylim([-1,1])
  plt.yticks([-1,-0.5,0,0.5,1])
  plt.suptitle(name)
  plt.savefig(base_dir+'/plots/e_cal-e_uncal-{0}.png'.format(name), dpi=300, bbox_inches='tight')

  plt.figure(2, figsize=(2*4.5, 3.75))
  plt.subplot(121)
  mode_uncal = np.sqrt(cat['e1_uncalibrated_supercals']**2. + cat['e2_uncalibrated_supercals']**2.)
  mode_cal = np.sqrt(cat['e1_calibrated_supercals']**2. + cat['e2_calibrated_supercals']**2.)
  plt.hist(mode_uncal, histtype='step', label='Uncalibrated')
  plt.hist(mode_cal, histtype='step', label='Calibrated')
  plt.xlabel('$|e|$')
  plt.legend()
  plt.subplot(122)
  pa_uncal = np.arctan2(cat['e2_uncalibrated_supercals'], cat['e1_uncalibrated_supercals'])
  pa_cal = np.arctan2(cat['e2_calibrated_supercals'], cat['e1_calibrated_supercals'])
  plt.hist(pa_uncal, histtype='step', label='Uncalibrated')
  plt.hist(pa_cal, histtype='step', label='Calibrated')
  plt.xlabel('PA [rad]')
  plt.suptitle(name)
  plt.savefig(base_dir+'/plots/mode-pa-{0}.png'.format(name), dpi=300, bbox_inches='tight')

  plt.figure(3, figsize=(2*4.5, 3.75))
  plt.subplot(121)
  plt.hist(cat['e1_uncalibrated_supercals'], histtype='step', label='Uncalibrated')
  plt.hist(cat['e1_calibrated_supercals'], histtype='step', label='Calibrated')
  plt.xlabel('$e_{1}$')
  plt.xlim([-1,1])
  plt.axvline(0, linestyle='dashed', color='k')#, alpha=0.4)
  plt.legend()
  plt.subplot(122)
  plt.hist(cat['e2_uncalibrated_supercals'], histtype='step', label='Uncalibrated')
  plt.hist(cat['e2_calibrated_supercals'], histtype='step', label='Calibrated')
  plt.xlim([-1,1])
  plt.axvline(0, linestyle='dashed', color='k')#, alpha=0.4)
  plt.xlabel('$e_{2}$')
  plt.suptitle(name)
  plt.savefig(base_dir+'/plots/e1_e2-{0}.png'.format(name), dpi=300, bbox_inches='tight')