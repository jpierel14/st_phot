"""
=============
Planting PSFs
=============
Planting a PSF with st_phot.
"""
	
###############################################################
# An example JWST Dataset is downloaded, and then we plant a 
# psf. This is public HST data for the
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
# **Download some Data**
#
# For this example we download JWST cal images from MAST. We just use
# 1 of the 8 dithered exposures for speed here.
obs_table = Observations.query_criteria(obs_id='jw02767-o002_t001_nircam_clear-f150w')
data_products_by_obs = Observations.get_product_list(obs_table)
data_products_by_obs = data_products_by_obs[data_products_by_obs['obs_id']=='jw02767002001_02103_00001_nrcb3']
data_products_by_obs = data_products_by_obs[data_products_by_obs['calib_level']==2]
data_products_by_obs = data_products_by_obs[data_products_by_obs['productSubGroupDescription']=='CAL']
Observations.download_products(data_products_by_obs,extension='fits')

####################################################################
# **Get the PSF model**
# 
# st_phot uses WebbPSF models for JWST. This can be pretty slow, 
# so you don't want to run this every time. Either create your
# own repository of these and pass each one when needed directly to
# the psf_photometry function, or else at least just do this once,
# save the ouptut, and then read it in and proceed to photometry
# for testing purposes.

files = glob.glob('mastDownload/JWST/jw02767002001_02103_00001_nrcb3/*cal.fits')
print(files)
plant_location = SkyCoord('21:29:42.4275','+0:04:53.634',unit=(u.hourangle,u.deg))
jwst_obs = st_phot.observation(files)
psfs = st_phot.get_jwst_psf(jwst_obs,plant_location,num_psfs=4,psf_width=9)
plt.imshow(psfs[0].data)
plt.show()

####################################################################
# **Examine the first Image**
# 
# You can see we've chosen a region of the image with no sources.

plant_image = files[0]
plant_fits = fits.open(plant_image)
plant_data = fits.open(plant_image)['SCI',1].data

plant_y,plant_x = skycoord_to_pixel(plant_location,wcs.WCS(plant_fits['SCI',1],plant_fits))
plant_cutout = extract_array(plant_data,(9,9),(plant_x,plant_y))
plt.imshow(plant_cutout, origin='lower')

plt.gca().tick_params(labelcolor='none',axis='both',color='none')
plt.show()



####################################################################
# **Plant the PSF**
# 

jwst_obs.plant_psf(psfs[0],[[plant_x,plant_y]],26)
planted_image = plant_image.replace('.fits','_plant.fits')
planted_data = fits.open(planted_image)['SCI',1].data
planted_cutout = extract_array(planted_data,(9,9),(plant_x,plant_y))

fig,axes = plt.subplots(1,2)
axes[0].imshow(plant_cutout, origin='lower')
axes[0].set_title('Pre-Plant')
axes[1].imshow(planted_cutout, origin='lower')
axes[1].set_title('Post-Plant')

for i in range(2):
    axes[i].tick_params(labelcolor='none',axis='both',color='none')
plt.show()


####################################################################
# **Measure PSF photometry and Aperture photometry for the source**
# 
jwst_obs = st_phot.observation(glob.glob('mastDownload/JWST/jw02767002001_02103_00001_nrcb3/*plant.fits')
)

jwst_obs.psf_photometry(psfs,plant_location,bounds={'flux':[-3000,100],
                        'centroid':[-1,1],
                        'bkg':[0,50]},
                        fit_width=5,
                        fit_bkg=True,
                        fit_flux='single')
jwst_obs.plot_psf_fit()
plt.show()

jwst_obs.plot_psf_posterior(minweight=.0005)
plt.show()

print('PSF Mag:',jwst_obs.psf_result.phot_cal_table['mag'])

jwst_obs.aperture_photometry(plant_location,encircled_energy='70')
print('Aperture Mag:',jwst_obs.aperture_result.phot_cal_table['mag'])