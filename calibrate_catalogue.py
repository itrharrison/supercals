import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import os
import configparser
from scipy.optimize import curve_fit
from numpy.polynomial import polynomial as polyn
import pickle

from astropy.table import Table, hstack, Column

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

from bias_surface import *

def poly_2d(x, y, deg=(1,1), coeffs=1.):

  terms = polyn.polyvander2d(x, y, deg)
  retVar = coeffs*terms

  return retVar

def polyfit_2d(x_obs, x_true, y_obs, y_true, deg):

  delta_x = x_obs - x_true
  delta_y = y_obs - y_true

  poly_x = poly_2d(x_true, y_true, deg=deg)
  poly_y = poly_2d(y_true, x_true, deg=deg)

  coeff_x, r_x, rank_x, s_x = np.linalg.lstsq(poly_x, delta_x)
  coeff_y, r_y, rank_y, s_y = np.linalg.lstsq(poly_y, delta_y)

  #coeff_x[0] += 0.3
  #coeff_y[0] += -0.13

  xnew = np.linspace(-0.7,0.7,128)
  ynew = np.linspace(-0.7,0.7,128)

  xx, yy = np.meshgrid(xnew, ynew)

  #b_x = np.sum(poly_2d(xx, yy, deg=deg, coeffs=coeff_x), axis=-1)
  #b_y = np.sum(poly_2d(yy, xx, deg=deg, coeffs=coeff_y), axis=-1)
  b_x = coeff_x
  b_y = coeff_y

  x_corrected = x_obs - np.sum(poly_2d(x_obs, y_obs, deg=deg, coeffs=coeff_x), axis=-1)
  y_corrected = y_obs - np.sum(poly_2d(y_obs, x_obs, deg=deg, coeffs=coeff_y), axis=-1)

  return x_corrected, y_corrected, b_x, b_y

def calibrate_supercals_catalogue(config, truth_cat_fname=None, doplots=False):

  new_columns = ['radius_im3shape',
                'e1_uncalibrated_supercals',
                'e2_uncalibrated_supercals',
                'e1_calibrated_supercals',
                'e2_calibrated_supercals',
                'e1_calibrated_supercals_1d',
                'e2_calibrated_supercals_1d',
                #'e1_corrected',
                #'e2_corrected',
                'e1_calibrated_iterative',
                'e2_calibrated_iterative']

  #cat_fname = 'level2-jvla-indiv-K27/cats/K27.tclean.image.tt0_split_000.srl.resolved.fits'
  #supercals_cat_fname = 'level2-jvla-indiv-K27/supercals/K27-uncalibrated-shape-catalogue.fits'
  base_dir = config.get('catalogues', 'base_dir')
  cat_fname = config.get('catalogues', 'wl_catalogue')
  supercals_cat_fname = config.get('catalogues', 'supercals_catalogue')
  supercals_dir = config.get('catalogues', 'supercals_directory')
  cat = Table.read(cat_fname)
  supercals_cat = Table.read(supercals_cat_fname)

  print(len(cat))

  #pdb.set_trace()

  for col in new_columns:
    cat[col] = np.nan
  cat['Valid_supercals'] = True

  cat['e1_corrected'] = Column(dtype=dict, length=len(cat))
  cat['e2_corrected'] = Column(dtype=dict, length=len(cat))
  cat['d_initial'] = Column(dtype=float, length=len(cat))
  cat['d_moved'] = Column(dtype=float, length=len(cat))
  cat['d_residual'] = Column(dtype=float, length=len(cat))

  for i,src in enumerate(cat):

    supercals_src = supercals_cat[np.argwhere(supercals_cat['Source_id']==str(src['Source_id']))]

    if len(supercals_src)==0:
      print('calibration missing!')
      continue

    if len(supercals_src) > 1:
      print('multiple calibrations! {0}'.format(supercals_src['Source_id']))
      supercals_src = supercals_src[0]


    calibration_cat_fname = supercals_dir+'/{0}_supercals.fits'.format(src['Source_id'])
    try:
      calibration_cat = Table.read(calibration_cat_fname)
    except FileNotFoundError:
      continue

    e1_bias_surface = bias_surface_2d(calibration_cat['e1_obs'], calibration_cat['e1_inp'], calibration_cat['e1_inp'], calibration_cat['e2_inp'])
    e2_bias_surface = bias_surface_2d(calibration_cat['e2_obs'], calibration_cat['e2_inp'], calibration_cat['e1_inp'], calibration_cat['e2_inp'])

    src['e1_uncalibrated_supercals'] = supercals_src['e1_obs']
    src['e2_uncalibrated_supercals'] = supercals_src['e2_obs']

    src['e1_calibrated_supercals'] = supercals_src['e1_obs'] - e1_bias_surface.ev(supercals_src['e1_obs'], supercals_src['e2_obs'])
    src['e2_calibrated_supercals'] = supercals_src['e2_obs'] - e2_bias_surface.ev(supercals_src['e1_obs'], supercals_src['e2_obs'])

    m_e1, c_e1 = bias_surface_1d(calibration_cat['e1_inp'], calibration_cat['e1_obs'])
    m_e2, c_e2 = bias_surface_1d(calibration_cat['e2_inp'], calibration_cat['e2_obs'])

    src['e1_calibrated_supercals_1d'] = (supercals_src['e1_obs'] - c_e1)/(1.e0 + m_e1)
    src['e2_calibrated_supercals_1d'] = (supercals_src['e2_obs'] - c_e2)/(1.e0 + m_e2)

    e1_corrected, e2_corrected, b_e1_0, b_e2_0 = polyfit_2d(calibration_cat['e1_obs'], calibration_cat['e1_inp'],
                                                            calibration_cat['e2_obs'], calibration_cat['e2_inp'],
                                                            deg=(0,0))

    e1_corrected, e2_corrected, b_e1_1, b_e2_1 = polyfit_2d(e1_corrected, calibration_cat['e1_inp'],
                                                            e2_corrected, calibration_cat['e2_inp'],
                                                            deg=(1,1))

    e1_corrected, e2_corrected, b_e1_2, b_e2_2 = polyfit_2d(e1_corrected, calibration_cat['e1_inp'],
                                                            e2_corrected, calibration_cat['e2_inp'],
                                                            deg=(2,2))

    e1_sample = supercals_src['e1_obs']
    e2_sample = supercals_src['e2_obs']

    e1_sample_c0 = e1_sample - np.sum(poly_2d(e1_sample, e2_sample, deg=(0,0), coeffs=b_e1_0), axis=-1)
    e2_sample_c0 = e2_sample - np.sum(poly_2d(e2_sample, e1_sample, deg=(0,0), coeffs=b_e2_0), axis=-1)

    e1_sample_c1 = e1_sample_c0 - np.sum(poly_2d(e1_sample_c0, e2_sample_c0, deg=(1,1), coeffs=b_e1_1), axis=-1)
    e2_sample_c1 = e2_sample_c0 - np.sum(poly_2d(e2_sample_c0, e1_sample_c0, deg=(1,1), coeffs=b_e2_1), axis=-1)

    e1_sample_c2 = e1_sample_c1 - np.sum(poly_2d(e1_sample_c1, e2_sample_c1, deg=(2,2), coeffs=b_e1_2), axis=-1)
    e2_sample_c2 = e2_sample_c1 - np.sum(poly_2d(e2_sample_c1, e1_sample_c1, deg=(2,2), coeffs=b_e2_2), axis=-1)


    src['e1_corrected'] = {'data' : e1_corrected}
    src['e2_corrected'] = {'data' : e2_corrected}
    src['e1_calibrated_iterative'] = e1_sample_c2
    src['e2_calibrated_iterative'] = e2_sample_c2


    src['radius_im3shape'] = supercals_src['radius']

    if doplots:

      e1_caled = src['e1_corrected']['data']
      e2_caled = src['e2_corrected']['data']

      # some distances
      distance_moved = np.sum(np.sqrt( (calibration_cat['e1_obs'] - e1_caled)**2. + (calibration_cat['e2_obs'] - e2_caled)**2. ))
      distance_from_true = np.sum(np.sqrt( (calibration_cat['e1_inp'] - e1_caled)**2. + (calibration_cat['e2_inp'] - e2_caled)**2. ))
      distance_initial = np.sum(np.sqrt( (calibration_cat['e1_inp'] - calibration_cat['e1_obs'])**2. + (calibration_cat['e2_inp'] - calibration_cat['e2_obs'])**2. ))

      src['d_initial'] = distance_initial
      src['d_moved'] = distance_moved
      src['d_residual'] = distance_from_true

      print(src['Source_id'])
      bias_surface_plots(calibration_cat['e1_obs'], calibration_cat['e1_inp'], calibration_cat['e1_inp'], calibration_cat['e2_inp'], e1_bias_surface, obs_point=[supercals_src['e1_obs'], supercals_src['e2_obs']], plot_fname=base_dir+'/src_{0}_e1_surface.png'.format(str(src['Source_id'])))
      bias_surface_plots(calibration_cat['e2_obs'], calibration_cat['e2_inp'], calibration_cat['e1_inp'], calibration_cat['e2_inp'], e2_bias_surface, obs_point=[supercals_src['e1_obs'], supercals_src['e2_obs']] ,plot_fname=base_dir+'/src_{0}_e2_surface.png'.format(str(src['Source_id'])))

      #calibration_cat = src_calibration_cat[src_calibration_cat['pointing']==ptg]
      plt.close('all')
      plt.figure(1, figsize=(4.5, 3.75))
      plt.scatter(calibration_cat['e1_obs'], calibration_cat['e2_obs'], edgecolors='powderblue', facecolors='none')
      #e1_caled = calibration_cat['e1_obs'] - e1_bias_surface.ev(calibration_cat['e1_obs'], calibration_cat['e2_obs'])
      #e2_caled = calibration_cat['e2_obs'] - e2_bias_surface.ev(calibration_cat['e1_obs'], calibration_cat['e2_obs'])
      plt.scatter(e1_caled, e2_caled, c='powderblue')
      plt.legend(['Uncalibrated', 'Calibrated'])
      plt.axvline(0, color='k', linestyle='dashed', zorder=-1)
      plt.axhline(0, color='k', linestyle='dashed', zorder=-1)
      plt.scatter(supercals_src['e1_obs'], supercals_src['e2_obs'], edgecolors='lightcoral', facecolors='none')
      plt.scatter(src['e1_calibrated_iterative'], src['e2_calibrated_iterative'], c='lightcoral')
      plt.plot([supercals_src['e1_obs'], src['e1_calibrated_iterative']], [supercals_src['e2_obs'], src['e2_calibrated_iterative']], '-', color='lightcoral')
      plt.xlim([-1,1])
      plt.ylim([-1,1])
      plt.xlabel('$e_1$')
      plt.ylabel('$e_2$')

      distances_string = 'Initial $ = {0:.2f}$'.format(distance_initial)
      distances_string = distances_string+'\n Moved $ = {0:.2f}$'.format(distance_moved)
      distances_string = distances_string+'\n Residual $= {0:.2f}$'.format(distance_from_true)
      print(distances_string)

      plt.suptitle(str(src['Source_id'])+'\n'+distances_string)
      plt.savefig(base_dir+'/{0}_cross_in_pointings.png'.format(str(src['Source_id'])), dpi=300, bbox_inches='tight')

      cross_data = {'e1_uncaled' : calibration_cat['e1_obs'],
                    'e2_uncaled' : calibration_cat['e2_obs'],
                    'e1_caled' : e1_caled,
                    'e2_caled' : e2_caled,
                    'src_e1_uncaled' : supercals_src['e1_obs'],
                    'src_e2_uncaled' : supercals_src['e2_obs'],
                    'src_e1_caled' : src['e1_calibrated_iterative'],
                    'src_e2_caled' : src['e2_calibrated_iterative']
                    }

      pickle.dump(cross_data, open(base_dir+'/{0}_cross_in_pointings.p'.format(str(src['Source_id'])), 'wb'))

  plt.close('all')
  plt.figure(1, figsize=(4.5, 3.75))
  plt.hist(cat['d_initial'], histtype='step', label='Initial', bins=25)
  plt.hist(cat['d_moved'], histtype='step', label='Moved', bins=25)
  plt.hist(cat['d_residual'], histtype='step', label='Residual', bins=25)
  plt.legend()
  plt.yscale('log')
  plt.xlabel('Calibration Cross Total Distance')
  plt.savefig(base_dir+'/cross_distances.png', bbox_inches='tight', dpi=300)

  print(len(cat))
  print(np.sum(cat['d_initial']>8))
  print(np.sum(cat['d_moved']>8))
  print(np.sum(cat['d_residual']>8))

  #pdb.set_trace()

  badlist = []

  for src in cat:
    if src['Source_id'] in badlist:
      src['Valid_supercals'] = False
      print('Excluded on badlist')
    #if np.sqrt(src['e1_calibrated_supercals']**2. + src['e2_calibrated_supercals']**2.) > 1:
    #  src['Valid_supercals'] = False
    #  print('Excluded on |e|>1')
    if np.isnan(src['e1_calibrated_supercals']):
      src['Valid_supercals'] = False
      print('Excluded on nan')
    if np.greater(src['d_residual'], 2.):
      src['Valid_supercals'] = False
      print('Excluded on cross failure')

  valid_cat = cat[cat['Valid_supercals']]

  make_cat_calibration_plots(valid_cat, base_dir=base_dir)#, name=cat_fname.split('/')[-1].split('.')[0])

  valid_cat_fname = base_dir+supercals_cat_fname.split('/')[-1].replace('.fits', '.supercals-calibrated.fits')
  print('Writing output catalogue to {0}'.format(valid_cat_fname))
  valid_cat.remove_columns(['e1_corrected', 'e2_corrected'])
  valid_cat.write(valid_cat_fname, overwrite=True)



  if truth_cat_fname is not None:
    
    from astropy.coordinates import SkyCoord
    
    truth_cat = Table.read(truth_cat_fname)

    coo_truth = SkyCoord(truth_cat['RA'], truth_cat['DEC'])
    coo_valid = SkyCoord(valid_cat['RA'], valid_cat['DEC'])

    idx_valid, d2d_valid, d3d_valid = coo_truth.match_to_catalog_sky(coo_valid)
    idx_truth, d2d_truth, d3d_truth = coo_valid.match_to_catalog_sky(coo_truth)

    valid_matches = idx_valid[d2d_valid.arcsec < 1.0]
    truth_matches = idx_truth[d2d_truth.arcsec < 1.0]

    valid_cat_matched = valid_cat[valid_matches]
    truth_cat_matched = truth_cat[truth_matches]

    match_cat = hstack([valid_cat_matched, truth_cat_matched])

    match_cat = match_cat[match_cat['radius_im3shape']>3]

    match_cat_fname = cat_fname.rstrip('.fits') + '.supercals-calibrated-withtruth.fits'
    match_cat.write(match_cat_fname, overwrite=True)

    print('match_cat {0}'.format(len(match_cat)))

    pybdsf_a = match_cat['DC_Maj']
    pybdsf_b = match_cat['DC_Min']
    pybdsf_mode = (pybdsf_a - pybdsf_b)/(pybdsf_a + pybdsf_b)
    pybdsf_pa = match_cat['PA_1'] - np.pi/2.
    pybdsf_e1 = pybdsf_mode*np.cos(2.*pybdsf_pa)
    pybdsf_e2 = pybdsf_mode*np.sin(2.*pybdsf_pa)

    plt.close('all')
    plt.figure(1, figsize=(2*4.5, 3.75))
    plt.subplot(121)
    plt.plot([-1,1],[0,0], 'k--', alpha=0.4)
    plt.plot(match_cat['e1'], match_cat['e1_calibrated_supercals']-match_cat['e1'], 'o')
    #plt.plot(match_cat['e1'], match_cat['e1_calibrated_supercals'], 'o')
    #plt.plot(match_cat['e1'], pybdsf_e1-match_cat['e1'], '+')
    plt.plot(match_cat['e1'], pybdsf_e1, '+')
    plt.xlabel('$e^{\\rm inp}_1$')
    plt.ylabel('$e^{\\rm meas}_1-e^{\\rm inp}_1$')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.subplot(122)
    plt.plot([-1,1],[0,0], 'k--', alpha=0.4)
    plt.xlabel('$e^{\\rm inp}_2$')
    #plt.ylabel('$e^{\\rm meas}_2$')
    plt.xlim([-1,1])
    plt.ylim([-1,1])
    plt.plot(match_cat['e2'], match_cat['e2_calibrated_supercals']-match_cat['e2'], 'o')
    #plt.plot(match_cat['e2'], match_cat['e2_calibrated_supercals'], 'o')
    plt.plot(match_cat['e2'], pybdsf_e2, '+')
    #plt.suptitle(cat_fname.split('/')[-4])
    plt.savefig(base_dir+'/plots/{0}-ein-eout.png'.format(supercals_cat_fname), dpi=300, bbox_inches='tight')

    np.savetxt(base_dir+'/supercals/{0}-ein-eout.txt'.format(supercals_cat_fname), np.column_stack([match_cat['e1'], match_cat['e1_calibrated_supercals'], match_cat['e1_uncalibrated_supercals'], pybdsf_e1, match_cat['radius_im3shape']]))


if __name__=='__main__':
  
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])

  calibrate_supercals_catalogue(config)
