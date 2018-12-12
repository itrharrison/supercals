import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
import os
import configparser

from astropy.table import Table, hstack

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots

from bias_surface import *

def calibrate_supercals_catalogue(config, truth_cat_fname=None):

  new_columns = ['radius_im3shape',
                'e1_uncalibrated_supercals',
                'e2_uncalibrated_supercals',
                'e1_calibrated_supercals',
                'e2_calibrated_supercals']

  #cat_fname = 'level2-jvla-indiv-K27/cats/K27.tclean.image.tt0_split_000.srl.resolved.fits'
  #supercals_cat_fname = 'level2-jvla-indiv-K27/supercals/K27-uncalibrated-shape-catalogue.fits'
  base_dir = config.get('catalogues', 'base_dir')
  cat_fname = config.get('catalogues', 'wl_catalogue')
  supercals_cat_fname = config.get('catalogues', 'supercals_catalogue')
  supercals_dir = config.get('catalogues', 'supercals_directory')
  cat = Table.read(cat_fname)
  supercals_cat = Table.read(supercals_cat_fname)

  print(len(cat))

  for col in new_columns:
    cat[col] = np.nan
  cat['Valid_supercals'] = True

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
    src['radius_im3shape'] = supercals_src['radius']

  badlist = []

  for src in cat:
    if src['Source_id'] in badlist:
      src['Valid_supercals'] = False
      print('Excluded on badlist')
    if np.sqrt(src['e1_calibrated_supercals']**2. + src['e2_calibrated_supercals']**2.) > 1:
      src['Valid_supercals'] = False
      print('Excluded on |e|>1')
    if np.isnan(src['e1_calibrated_supercals']):
      src['Valid_supercals'] = False
      print('Excluded on nan')

  valid_cat = cat[cat['Valid_supercals']]

  make_cat_calibration_plots(valid_cat, base_dir=base_dir, name=base_dir[-3:])

  valid_cat_fname = cat_fname.rstrip('.fits') + '.supercals-calibrated.fits'

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
    plt.plot(match_cat['e1'], pybdsf_e1-match_cat['e1'], '+')
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
    plt.plot(match_cat['e2'], match_cat['e2_calibrated_supercals']-match_cat['e1'], 'o')
    plt.suptitle(cat_fname.split('/')[2])
    plt.savefig(base_dir+'/plots/ein-eout.png', dpi=300, bbox_inches='tight')

    np.savetxt('/Users/harrison/Dropbox/code_mcr/supercals/{0}-ein-eout.txt'.format(base_dir[-3:]), np.column_stack([match_cat['e1'], match_cat['e1_calibrated_supercals'], match_cat['e1_uncalibrated_supercals'], pybdsf_e1, match_cat['radius_im3shape']]))


if __name__=='__main__':
  
  config = ConfigParser.ConfigParser()
  config.read(sys.argv[-1])

  calibrate_supercals_catalogue(config)
