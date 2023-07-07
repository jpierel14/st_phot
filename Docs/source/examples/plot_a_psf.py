"""
==============
PSF Photometry
==============
Measuring PSF Photometry with st_phot.
"""
	
###############################################################
# An example HST Dataset is downloaded, and then we measure 
# psf photometry. This is public HST data for the
# gravitationally lensed SN 2022riv
   

import sys,os,glob
from astropy.io import fits
from astropy.table import Table
from astropy.nddata import extract_array
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.wcs.utils import skycoord_to_pixel
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astropy.visualization import (simple_norm,LinearStretch)

import st_phot


####################################################################
# 
# ----------
# HST Images
# ----------
#
# **Download some Data**
#
# For this example we download HST FLT images from MAST.  

obs_table = Observations.query_criteria(obs_id='hst_16264_12_wfc3_ir_f110w_iebc12')
obs_table1 = obs_table[obs_table['filters']=='F110W']

data_products_by_obs = Observations.get_product_list(obs_table1)
data_products_by_obs = data_products_by_obs[data_products_by_obs['calib_level']==2]
data_products_by_obs = data_products_by_obs[data_products_by_obs['productSubGroupDescription']=='FLT'][:3]
Observations.download_products(data_products_by_obs,extension='fits')


####################################################################
# **Examine the first Image**
# 

files = glob.glob('mastDownload/HST/*/*flt.fits')
ref_image = files[0]
ref_fits = fits.open(ref_image)
ref_data = fits.open(ref_image)['SCI',1].data
norm1 = simple_norm(ref_data,stretch='linear',min_cut=-1,max_cut=10)

plt.imshow(ref_data, origin='lower',
                      norm=norm1,cmap='gray')
plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()

####################################################################
# **Zoom in to see the Supernova**
# 

sn_location = SkyCoord('21:29:40.2110','+0:05:24.154',unit=(u.hourangle,u.deg))
ref_y,ref_x = skycoord_to_pixel(sn_location,wcs.WCS(ref_fits['SCI',1],ref_fits))
ref_cutout = extract_array(ref_data,(11,11),(ref_x,ref_y))
norm1 = simple_norm(ref_cutout,stretch='linear',min_cut=-1,max_cut=10)
plt.imshow(ref_cutout, origin='lower',
                      norm=norm1,cmap='gray')
plt.title('SN2022riv')
plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()

####################################################################
# **Get the PSF model**
# 
# st_phot uses Jay Anderson's gridded HST PSF models. Some filters
# are missing, so for those you'll have to either use a 
# neighboring filter or build your own PSF from stars in the field.

hst_obs = st_phot.observation2(files)
psfs = st_phot.get_hst_psf(hst_obs,sn_location)
plt.imshow(psfs[0].data)
plt.show()

####################################################################
# **Measure the PSF photometry**
# 
hst_obs.psf_photometry(psfs,sn_location,bounds={'flux':[-3000,100],
                        'centroid':[-.5,.5],
                        'bkg':[0,10]},
                        fit_width=5,
                        fit_bkg=True,
                        fit_flux='single')
hst_obs.plot_psf_fit()
plt.show()

hst_obs.plot_psf_posterior(minweight=.0005)
plt.show()

print(hst_obs.psf_result.phot_cal_table)

####################################################################
# **Flux per exposure**
# 
# You can also fit for a flux in every exposure, instead of a single
# flux across all exposures
hst_obs.psf_photometry(psfs,sn_location,bounds={'flux':[-3000,100],
                        'centroid':[-.5,.5],
                        'bkg':[0,10]},
                        fit_width=5,
                        fit_bkg=True,
                        fit_flux='multi')
hst_obs.plot_psf_fit()
plt.show()

hst_obs.plot_psf_posterior(minweight=.0005)
plt.show()

print(hst_obs.psf_result.phot_cal_table)

####################################################################
# 
# -----------
# JWST Images
# -----------
#
# **Download some Data**
#
# For this example we download JWST cal images from MAST. We just use
# 4 of the 8 dithered exposures  for speed here, but in principle
# st_phot can handle as many as are needed (given time).
obs_table = Observations.query_criteria(obs_id='jw02767-o002_t001_nircam_clear-f150w')
data_products_by_obs = Observations.get_product_list(obs_table)
data_products_by_obs = data_products_by_obs[data_products_by_obs['calib_level']==2]
data_products_by_obs = data_products_by_obs[data_products_by_obs['productSubGroupDescription']=='CAL']

# Just take the nrcb3 cals (where the SN is located)
to_remove = []
for i in range(len(data_products_by_obs)):
    if not data_products_by_obs[i]['obs_id'].endswith('nrcb3'):
        to_remove.append(i)
data_products_by_obs.remove_rows(to_remove)
Observations.download_products(data_products_by_obs[:4],extension='fits')

####################################################################
# **Examine the first Image**
# 

files = glob.glob('mastDownload/JWST/*/*cal.fits')
ref_image = files[0]
ref_fits = fits.open(ref_image)
ref_data = fits.open(ref_image)['SCI',1].data
norm1 = simple_norm(ref_data,stretch='linear',min_cut=-1,max_cut=10)

plt.imshow(ref_data, origin='lower',
                      norm=norm1,cmap='gray')
plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()

####################################################################
# **Zoom in to see the Supernova**
# 

sn_location = SkyCoord('21:29:40.2103','+0:05:24.158',unit=(u.hourangle,u.deg))
ref_y,ref_x = skycoord_to_pixel(sn_location,wcs.WCS(ref_fits['SCI',1],ref_fits))
ref_cutout = extract_array(ref_data,(11,11),(ref_x,ref_y))
norm1 = simple_norm(ref_cutout,stretch='linear',min_cut=-1,max_cut=10)
plt.imshow(ref_cutout, origin='lower',
                      norm=norm1,cmap='gray')
plt.title('SN2022riv')
plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()

####################################################################
# **Get the PSF model**
# 
# st_phot uses WebbPSF models for JWST. This can be pretty slow, 
# so you don't want to run this every time. Either create your
# own repository of these and pass each one when needed directly to
# the psf_photometry function, or else at least just do this once,
# save the ouptut, and then read it in and proceed to photometry
# for testing purposes.

jwst_obs = st_phot.observation2(files)
psfs = st_phot.get_jwst_psf(jwst_obs,sn_location,num_psfs=4)
plt.imshow(psfs[0].data)
plt.show()

####################################################################
# **Measure the PSF photometry**
# 
jwst_obs.psf_photometry(psfs,sn_location,bounds={'flux':[-1000,1000],
                        'centroid':[-2,2],
                        'bkg':[0,50]},
                        fit_width=5,
                        fit_bkg=True,
                        fit_flux='single')
jwst_obs.plot_psf_fit()
plt.show()

jwst_obs.plot_psf_posterior(minweight=.0005)
plt.show()

print(jwst_obs.psf_result.phot_cal_table)

#####################################################################
# 
# -----------
# Level 3 PSF
# -----------
#
# While it's generally recommended to perform PSF photometry on data
# with level 2 processing (i.e., before drizzling), sometimes low
# S/N means it's desirable to perform PSF photometry on level 3 data.
# While (usually) not quite as accurate, here is a function to do
# this. 
#
# **Download some Data**
#
# For this example we download JWST cal images from MAST. We just use
# 4 of the 8 dithered exposures  for speed here, but in principle
# st_phot can handle as many as are needed (given time).
obs_table = Observations.query_criteria(obs_id='jw02767-o002_t001_nircam_clear-f150w')
data_products_by_obs = Observations.get_product_list(obs_table)
data_products_by_obs = data_products_by_obs[data_products_by_obs['calib_level']==3]
data_products_by_obs = data_products_by_obs[data_products_by_obs['productSubGroupDescription']=='I2D']
Observations.download_products(data_products_by_obs[0],extension='fits')

####################################################################
# **Examine the Image**
# 

files = glob.glob('mastDownload/JWST/*/*i2d.fits')
ref_image = files[0]
ref_fits = fits.open(ref_image)
ref_data = fits.open(ref_image)['SCI',1].data
norm1 = simple_norm(ref_data,stretch='linear',min_cut=-1,max_cut=10)

plt.imshow(ref_data, origin='lower',
                      norm=norm1,cmap='gray')
plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()

####################################################################
# **Zoom in to see the Supernova**
# 

sn_location = SkyCoord('21:29:40.2103','+0:05:24.158',unit=(u.hourangle,u.deg))
ref_y,ref_x = skycoord_to_pixel(sn_location,wcs.WCS(ref_fits['SCI',1],ref_fits))
ref_cutout = extract_array(ref_data,(11,11),(ref_x,ref_y))
norm1 = simple_norm(ref_cutout,stretch='linear',min_cut=-1,max_cut=10)
plt.imshow(ref_cutout, origin='lower',
                      norm=norm1,cmap='gray')
plt.title('SN2022riv (level 3)')
plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()

####################################################################
# **Get the PSF model**
# 
# (note the use of "3" instead of "2" everywhere). And note that it
# is the level 2 observation, not 3, that is passed to the psf
# function. That is so the PSF model can be drizzled using the same
# pattern used to drizzle the data. You can do the same with HST
# by just replacing "jwst" with "hst". 

jwst3_obs = st_phot.observation3(files[0])
psf3 = st_phot.get_jwst3_psf(jwst_obs,sn_location,num_psfs=4)
plt.imshow(psf3.data)
plt.show()

####################################################################
# **Measure the PSF photometry**
# 
jwst3_obs.psf_photometry(psf3,sn_location,bounds={'flux':[-1000,1000],
                        'centroid':[-2,2],
                        'bkg':[0,50]},
                        fit_width=5,
                        fit_bkg=True,
                        fit_flux=True)
jwst3_obs.plot_psf_fit()
plt.show()

jwst3_obs.plot_psf_posterior(minweight=.0005)
plt.show()

