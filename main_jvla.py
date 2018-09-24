import numpy as np
import sys
import ConfigParser

from supercals import *
from supercals_postprocess import *

config = ConfigParser.ConfigParser()
config.read(sys.argv[-1])

wl_catalogue = config.get('survey', 'wl_catalogue')
pointing_rootdir = config.get('survey', 'pointing_root_directory')
output_cat_rootdir = config.get('output', 'output_cat_dir')
output_plot_rootdir = config.get('output', 'output_plot_dir')
pointing_list = config.get('survey', 'pointing_list').split(',')

for pointing in pointing_list:
  pointing_fname_root = pointing_rootdir+'/'+pointing+'/'+'jvla_dr1_'+pointing+'_I_B'
  config.set('input', 'catalogue', wl_catalogue)
  config.set('input', 'residual_image', pointing_fname_root+'.residual.tt0.fits')
  config.set('input', 'clean_image', pointing_fname_root+'.image.tt0.fits')
  config.set('input', 'psf_image', pointing_fname_root+'.image.tt0.fits')
  config.set('input', 'model_image', pointing_fname_root+'.model.tt0.fits')
    
    if not config.has_option('output', 'output_plot_dir_base'):
    plot_dir_base = config.get('output', 'output_plot_dir')
    config.set('output', 'output_plot_dir_base', plot_dir_base)

  if not config.has_option('output', 'output_cat_dir_base'):
    plot_dir_base = config.get('output', 'output_cat_dir')
    config.set('output', 'output_cat_dir_base', plot_dir_base)

  config.set('output', 'output_plot_dir', config.get('output', 'output_plot_dir_base')+'/'+pointing+'/plots/')
  config.set('output', 'output_cat_dir', config.get('output', 'output_cat_dir_base')+'/'+pointing+'/cats/')
  config.set('input', 'pointing_name', pointing)
  
  runSuperCal(config)
  # need to think about this
  # do we want to used averaged corrections on averaged measurements?
  # is this equivalent to averaging after correction (probably)?
  #for source in wl_cat:
  #  make_m_and_c(source, config)
  #  calculate_corrected_ellipticity(source_in_pointing, config)
  #  make_calib_surface_plots(config)

