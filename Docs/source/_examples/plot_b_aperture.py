"""
===================
Aperture Photometry
===================
Measuring PSF Photometry with st_phot.
"""
	
###############################################################
# An example HST Dataset is downloaded, and then we measure 
# aperture photometry. This is public HST data for the
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
# **Measure the aperture photometry**
# 
hst_obs = st_phot.observation(files)
hst_obs.aperture_photometry(sn_location,radius=3,
                    skyan_in=5,skyan_out=7)
print(hst_obs.aperture_result.phot_cal_table)


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
# **Measure the aperture photometry**
# 
jwst_obs = st_phot.observation(files)
jwst_obs.aperture_photometry(sn_location,encircled_energy='70')
print(jwst_obs.aperture_result.phot_cal_table)


