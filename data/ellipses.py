import copy
import numpy as np
from scipy import fftpack as ft
from scipy import ndimage as ndi
from matplotlib import pyplot as plt
from pylab import normpdf
from math import pi

def gaussian_profile(r, r0, I_0=1.e0):

  retVar = I_0*np.exp( - (r)**2 / (2 * r0**2) )

  return retVar

def sersic_profile(r, r0, I_0=1.e0, n=4.e0):
    
  k = 1.e0
  lnI = np.log(I_0) - k*pow(r/r0, 1.e0/n)
  
  return np.exp(lnI)

def tophat_profile(r, r0, I0=1.e0):
  
  retVar = np.zeros_like(r)
  retVar[r<r0] = I0
  
  return retVar
  
def r_sheared(xx, yy, shear_components):
  
  m1 = shear_components[0]
  m2 = shear_components[1]
  m3 = shear_components[2]
  
  return m1*xx*xx + 2.e0*m2*xx*yy + m3*yy*yy
  
def shear_matrix(e1, e2):

  if np.sqrt(e1*e1 + e2*e2) > 1:
    raise Exception("ERROR: Unphysical ellipticity")
  
  e = e1*e1 + e2*e2
  
  m1 = (1.e0 + e - 2.e0*e1)/(1.e0 - e)
  m2 = (-2.e0*e2)/(1.e0 - e)
  m3 = (1.e0 + e + 2.e0*e1)/(1.e0 - e)
  
  return np.array([[m1, m2/2.], [m2/2., m3]])
  
def shear_matrix_components(e1, e2):

  if np.sqrt(e1*e1 + e2*e2) > 1:
    raise Exception("ERROR: Unphysical ellipticity")
  
  e = e1*e1 + e2*e2
  
  m1 = (1.e0 + e - 2.e0*e1)/(1.e0 - e)
  m2 = (-2.e0*e2)/(1.e0 - e)
  m3 = (1.e0 + e + 2.e0*e1)/(1.e0 - e)
  
  return m1, m2, m3

def ellipse_abt(a, b, rotang, xc):
  
  t = np.linspace(0.e0, 2.e0*pi, 100)
    
  x = a*np.cos(t)*np.cos(rotang) - b*np.sin(t)*np.sin(rotang) + xc[0]
  y = a*np.cos(t)*np.sin(rotang) + b*np.sin(t)*np.cos(rotang) + xc[1]
  
  return x, y
  
def ellipse_e(e1, e2, xc, scale=0.03e0):
  
  e, a, b, rotang = e2ab(e1, e2, scale)
  
  return ellipse_abt(a, b, rotang, xc)
  
def e2ab(e1, e2, scale=0.03e0):
  
  e = np.sqrt(e1*e1 + e2*e2)
  a = (1.e0 + e)*scale
  b = (1.e0 - e)*scale
  rotang = 0.5e0*np.arctan2(e2,e1)
  
  return e, a, b, rotang
  
if __name__ == '__main__':
  
  """
  fig1 = plt.figure(1)
  f1a1 = fig1.add_subplot(111)
  
  erange = np.linspace(-1.e0, 1.e0, 17)
  for e1 in erange:
    for e2 in erange:
      e = np.sqrt(e1*e1 + e2*e2)
      if e > 1:
        continue
      x, y = ellipse_e(e1, e2, [e1, e2])
      f1a1.plot(x, y, 'b-',ms=1)
  """
  plt.close('all')
  Npix = 256
  xextent = [-5,5]
  yextent = [-5,5]

  field = np.zeros([Npix,Npix])
  x = np.linspace(xextent[0],xextent[1],Npix)
  y = np.linspace(yextent[0],yextent[1],Npix)
  
  xx, yy = np.meshgrid(x,y)
  r = np.sqrt(xx*xx + yy*yy)
  
  field += sersic_profile(r, 2.e0)

  sheared_field = field * shear_matrix(0.1, 0.4) * field.T
  
  plt.imshow(field)
  plt.show()

  '''
  fig2 = plt.figure(1)
  f2a1 = fig2.add_subplot(321)
  f2a2 = fig2.add_subplot(322)
  f2a3 = fig2.add_subplot(323)
  f2a4 = fig2.add_subplot(324)
  f2a5 = fig2.add_subplot(325)
  f2a6 = fig2.add_subplot(326)
  
  psf = np.zeros_like(field)
  gauss = normpdf(r,0,30.e0/Npix)
  
  f2a1.imshow(field)
  f2a2.imshow(gauss)

  #obs = ndi.convolve(field, gauss)
  
  F_field = (ft.fft2(field))
  F_gauss = (ft.fft2(gauss))
  
  F_obs = F_field*F_gauss
  
  obs = ft.fftshift(ft.ifft2(F_obs)).real
  
  F_obs = ft.fftshift(ft.fft2(obs))
  ampl = abs(F_obs)
  phas = np.arctan2(F_obs.imag, F_obs.real)
  
  f2a3.imshow(F_obs.real)
  f2a4.imshow(F_obs.imag)
  f2a5.imshow(ampl)
  f2a6.imshow(phas)
  
  fig2.show()
  
  fig3 = plt.figure(3)
  f3a1 = fig3.add_subplot(111)
  f3a1.imshow(obs.real)
  fig3.show()
  
  ampl_shuff = copy.copy(ampl)
  np.random.shuffle(ampl_shuff)
  
  F_obs_shuff = ampl_shuff*(np.cos(phas) + 1j*np.sin(phas))
  obs_shuff = ft.fftshift(ft.ifft2(F_obs_shuff))
  
  fig4 = plt.figure(4)
  f4a1 = fig4.add_subplot(111)
  f4a1.imshow(obs_shuff)
  fig4.show()
  '''
  
"""
def ellipse_outline(xc=[0.e0,0.e0], major=1, minor=0.5e0, angle=45):
  
  x = major*np.cos(t) - xc[0]
  y = minor*np.cos(t) - xc[1]
  
  theta = np.deg2rad(angle)
  rotmat = np.matrix([[cos(theta),sin(theta)],[-sin(theta),cos(theta)]])
  xy = np.matrix([x, y])
  
  xyp = xy.T*rotmat
  
  return xyp[:,0], xyp[:,1]  

def ellipse(xx, yy, xc=[0.e0,0.e0], major=1, minor=0.5e0):
  
  sl = (((xx-xc[0])/major)**2 + ((yy-xc[1])/minor)**2) < 1
  
  return sl

Npix = 128
xextent = [-5,5]
yextent = [-5,5]

major = 2.e0
minor = 1.e0

field = np.random.normal(0.e0, 5.e-2, [Npix,Npix])
x = np.linspace(xextent[0],xextent[1],Npix)
y = np.linspace(yextent[0],yextent[1],Npix)

xx, yy = np.meshgrid(x,y)
r = np.sqrt(xx**2.e0 + yy**2.e0)

#field += sersic(r)
field[ellipse(xx, yy)] += 1

fig1 = plt.figure(1)
f1a1 = fig1.add_subplot(221)
f1a2 = fig1.add_subplot(222)
f1a3 = fig1.add_subplot(223)
f1a1.imshow(field)

psf = np.zeros_like(field)
gauss = normpdf(r,0,30.e0/Npix)

f1a2.imshow(gauss)

obs = ndi.convolve(field, gauss)

f1a3.imshow(obs)

fig1.show()
"""
