import numpy as np
import sys
import configparser

from supercal import *
from supercal_postprocess import *

config = ConfigParser.ConfigParser()
config.read(sys.argv[-1])

wl_catalogue = config.get('survey', 'wl_catalogue')
pointing_list = config.get('survey', 'pointing_list')

for pointing in pointing_list:

  config.set('input', 'catalogue', wl_catalogue)
  config.set('input', 'residual_image', pointing+'-residual.fits')
  config.set('input', 'clean_image', pointing+'-image.fits')
  config.set('input', 'psf_image', pointing+'-psf.fits')
  config.set('output', 'output_cat_dir', pointing+'/supercal-output/')
  
  runSuperCal(config)
  # need to think about this
  # do we want to used averaged corrections on averaged measurements?
  # is this equivalent to averaging after correction (probably)?
  for source in wl_cat:
    make_m_and_c(source, config)
    calculate_corrected_ellipticity(source_in_pointing, config)
    make_calib_surface_plots(config)

