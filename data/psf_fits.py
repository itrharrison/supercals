import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from ellipses import *

from astropy import units as uns

rc('text', usetex=True)
rc('font', family='serif')
rc('font', size=11)

plt.close('all') # tidy up any unshown plots


from astropy.io import fits

#psf = fits.getdata('level2-emerlin-100times-composite-noagn-psf.fits')
#psf_header = fits.getheader('level2-emerlin-100times-composite-noagn-psf.fits')

psf = fits.getdata('/local/scratch/harrison/simuCLASS/level2-emerlin-100times-composite-noagn-uniform-psf.fits')
psf_header = fits.getheader('/local/scratch/harrison/simuCLASS/level2-emerlin-100times-composite-noagn-uniform-psf.fits')

npix = 32
psize = int(psf.shape[-1])

bmaj = psf_header['BMAJ']*uns.degree
bmin = psf_header['BMIN']*uns.degree
bpa = psf_header['BPA']*uns.degree

#bmaj = 0.05*uns.arcsec
#bmin = 0.04*uns.arcsec

q = bmin/bmaj
e = (1 - q)/(1 + q)
beta = bpa.to(uns.rad).value
e1 = e*np.cos(2.*beta)
e2 = e*np.sin(2.*beta)

pix_scale = abs(psf_header['CDELT1'])*uns.degree
pix_scale_arcsec = pix_scale.to(uns.arcsec).value
extent = np.linspace(-npix*pix_scale_arcsec, npix*pix_scale_arcsec, 2*npix)

psf_cen = psf[0,0,psize/2 - npix:psize/2 + npix, psize/2 - npix:psize/2 + npix]

clean_psf = np.zeros([2*npix,2*npix])
x = extent
y = extent

xx, yy = np.meshgrid(x,y)
r = np.sqrt(xx**2. + yy**2.)

shear_matrix = np.zeros([2,2])
shear_matrix[0,0] = 1 - e1
shear_matrix[0,1] = -e2
shear_matrix[1,0] = -e2
shear_matrix[1,1] = 1 + e1

xx_sheared = (1 - e1)*xx + (-e2)*yy
yy_sheared = (-e2)*xx + (1 + e1)*yy

rr_sheared = np.sqrt(xx_sheared**2 + yy_sheared**2)

#r = r_sheared(xx, yy, shear_matrix_components(e1, e2))

#clean_psf += sersic_profile(r, bmaj.to(uns.arcsec).value, n=0.5)
clean_psf += gaussian_profile(rr_sheared, bmaj.to(uns.arcsec).value)

plt.figure(1, figsize=(9, 7.5))
ax1 = plt.subplot(221)
plt.plot(extent, psf_cen[npix,:])
plt.plot(extent, clean_psf[npix,:])
plt.xlabel('$\mathrm{RA \, [arcsec]}$')
plt.subplot(224)
plt.plot(extent, psf_cen[:,npix], label='$\mathrm{Dirty}$')
plt.plot(extent, clean_psf[:,npix], label='$\mathrm{CLEAN}$')
plt.xlabel('$\mathrm{DEC \, [arcsec]}$')
plt.legend()
plt.subplot(222, sharex=ax1)
plt.imshow(clean_psf, extent=[-npix*pix_scale_arcsec, npix*pix_scale_arcsec, -npix*pix_scale_arcsec, npix*pix_scale_arcsec], aspect='equal', vmin=psf_cen.min(), vmax=psf_cen.max())
plt.axis('off')
ax3 = plt.subplot(223, sharex=ax1)
plt.imshow(psf_cen, extent=[-npix*pix_scale_arcsec, npix*pix_scale_arcsec, -npix*pix_scale_arcsec, npix*pix_scale_arcsec], aspect='equal', vmin=psf_cen.min(), vmax=psf_cen.max())
plt.axis('off')
plt.suptitle('$\mathrm{CASA \, Clean \, PSF}$', y=0.925)
plt.savefig('emerlin_clean_psf.png', bbox_inches='tight', dpi=320)
