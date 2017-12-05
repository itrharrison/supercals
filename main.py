import numpy as np
import sys
import ConfigParser

from supercal import *
from supercal_postprocess import *

config = ConfigParser.ConfigParser()
config.read(sys.argv[-1])

wl_catalogue = config.get('survey', 'wl_catalogue')
pointing_rootdir = config.get('survey', 'pointing_root_directory')
pointing_list = config.get('survey', 'pointing_list').split(',')

for pointing in pointing_list:
  pointing_fname_root = pointing_rootdir+'/'+pointing+'/'+pointing+'_Peeled_natw'
  config.set('input', 'catalogue', wl_catalogue)
  config.set('input', 'residual_image', pointing_fname_root+'-residual.fits')
  config.set('input', 'clean_image', pointing_fname_root+'-image.fits')
  config.set('input', 'psf_image', pointing_fname_root+'-psf.fits')
  config.set('output', 'output_cat_dir', pointing_fname_root+'/supercal-output/')
  config.set('input', 'pointing_name', pointing)
  
  runSuperCal(config)
  # need to think about this
  # do we want to used averaged corrections on averaged measurements?
  # is this equivalent to averaging after correction (probably)?
  #for source in wl_cat:
  #  make_m_and_c(source, config)
  #  calculate_corrected_ellipticity(source_in_pointing, config)
  #  make_calib_surface_plots(config)

