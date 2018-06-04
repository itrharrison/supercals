import numpy as np
import sys
import ConfigParser

from supercals import *
from postprocess import *

config = ConfigParser.ConfigParser()
config.read(sys.argv[-1])

wl_catalogue = config.get('survey', 'wl_catalogue')
pointing_rootdir = config.get('survey', 'pointing_root_directory')
output_cat_rootdir = config.get('output', 'output_cat_dir')
output_plot_rootdir = config.get('output', 'output_plot_dir')
pointing_list = config.get('survey', 'pointing_list').split(',')

for pointing in pointing_list:
  pointing_fname_root = pointing_rootdir+'/'+pointing+'/'+pointing+config.get('survey', 'postfix')
  config.set('input', 'catalogue', wl_catalogue)
  config.set('input', 'residual_image', pointing_fname_root+'-residual.fits')
  config.set('input', 'clean_image', pointing_fname_root+'-image.fits')
  config.set('input', 'psf_image', pointing_fname_root+'-psf.fits')
  config.set('input', 'model_image', pointing_fname_root+'-model.fits')

  if not config.has_option('output', 'output_plot_dir_base'):
    plot_dir_base = config.get('output', 'output_plot_dir')
    config.set('output', 'output_plot_dir_base', plot_dir_base)

  if not config.has_option('output', 'output_cat_dir_base'):
    plot_dir_base = config.get('output', 'output_cat_dir')
    config.set('output', 'output_cat_dir_base', plot_dir_base)

  config.set('output', 'output_plot_dir', config.get('output', 'output_plot_dir_base')+'/'+pointing+'/')
  config.set('output', 'output_cat_dir', config.get('output', 'output_cat_dir_base')+'/'+pointing+'/')
  config.set('input', 'pointing_name', pointing)
  
  if config.getboolean('pipeline', 'do_supercals'):
    runSuperCal(config)
  if config.getboolean('pipeline', 'do_calibration'):
    runCalibration(config)

if config.getboolean('pipeline', 'do_create_calibrated_catalogue'):
  runCreateCatalogue(config)
