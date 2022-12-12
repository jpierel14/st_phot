import warnings,os,sys,time,glob
import numpy as np
import urllib.request
import scipy
import webbpsf

import sncosmo

import astropy
from astropy import wcs
from astropy.io import fits
from astropy.table import Table,vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.time import Time

from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.psf import EPSFModel
warnings.simplefilter('ignore')
import jwst
from jwst import datamodels
from jwst import source_catalog
from jwst.source_catalog import reference_data

from .wfc3_photometry.psf_tools.PSFUtils import make_models
from .wfc3_photometry.psf_tools.PSFPhot import get_standard_psf

__all__ = ['get_jwst_psf','get_hst_psf']
def get_jwst_psf(st_obs,sky_location,num_psfs=16,psf_width=41):
    inst = webbpsf.instrument(st_obs.instrument)
    inst.filter = st_obs.filter
    inst.detector=st_obs.detector

    grid = inst.psf_grid(num_psfs=num_psfs,all_detectors=False)
    psf_list = []
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        grid.x_0 = x
        grid.y_0 = y
        yg, xg = np.mgrid[-1*(psf_width-1)/2:(psf_width+1)/2,-1*(psf_width-1)/2:(psf_width+1)/2].astype(int)
        yf, xf = yg+int(y+.5), xg+int(x+.5)
        psf = np.array(grid(xf,yf)).astype(float)
        epsf_model = EPSFModel(psf)
        psf_list.append(epsf_model)
    return psf_list

def get_hst_psf(st_obs,sky_location,psf_width=21):
    grid = make_models(get_standard_psf(os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'wfc3_photometry/psfs'),st_obs.filter,st_obs.detector))[0]
    psf_list = []
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        grid.x_0 = x
        grid.y_0 = y
        yg, xg = np.mgrid[-1*(psf_width-1)/2:(psf_width+1)/2,-1*(psf_width-1)/2:(psf_width+1)/2].astype(int)
        yf, xf = yg+int(y+.5), xg+int(x+.5)
        psf = np.array(grid(xf,yf)).astype(float)
        epsf_model = EPSFModel(psf)
        psf_list.append(epsf_model)
    return psf_list

def jwst_apcorr(fname,ee=70):
    sc = source_catalog.source_catalog_step.SourceCatalogStep()
    with datamodels.open(fname) as model:
        reffile_paths = sc._get_reffile_paths(model)
        aperture_ee = (20,30,ee)

        refdata = reference_data.ReferenceData(model, reffile_paths,
                                aperture_ee)
        aperture_params = refdata.aperture_params
    return [aperture_params['aperture_radii'][-1], 
           aperture_params['aperture_corrections'][-1],
           aperture_params['bkg_aperture_inner_radius'],
           aperture_params['bkg_aperture_outer_radius']]


def estimate_bkg(data,position,inner, outer,model_psf=None,corr=None):
    assert model_psf is not None or corr is not None, 'Must supply model_psf or corr'
    assert inner<outer

    annulus_aperture = CircularAnnulus(np.flip(position), r_in=inner, r_out=outer)
    annulus_mask = annulus_aperture.to_mask(method='center')

    annulus_data = annulus_mask.multiply(data)
    import matplotlib.pyplot as plt
    model_psf.x_0 = position[1]
    model_psf.y_0 = position[0]
    yf, xf = np.mgrid[0:data.shape[0],0:data.shape[1]].astype(int)
    psf = np.array(model_psf(xf,yf)).astype(float)
    annulus_psf = annulus_mask.multiply(psf)
    print(np.sum(annulus_psf)/np.sum(psf))
    plt.imshow(annulus_data)
    plt.show()
    plt.imshow(annulus_psf)
    plt.show()
    sys.exit()

def generic_aperture_phot(data,positions,radius,sky,epadu=1,error=None):
    aperture = CircularAperture(positions, r=radius)
    annulus_aperture = CircularAnnulus(positions, r_in=sky["sky_in"], r_out=sky["sky_out"])
    annulus_mask = annulus_aperture.to_mask(method='center')

    bkg_median = []
    bkg_stdev = []
    for mask in annulus_mask:
        annulus_data = mask.multiply(data)
        annulus_data_1d = annulus_data[mask.data > 0]
        _, median_sigclip, stdev_sigclip = sigma_clipped_stats(annulus_data_1d)
        bkg_median.append(median_sigclip)
        bkg_stdev.append(stdev_sigclip)
    bkg_median = np.array(bkg_median)#32.672334253787994#33#

    bkg_stdev = np.array(bkg_stdev)

    phot = aperture_photometry(data, aperture, method='exact',error=error)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    if error is None:
        error_poisson = np.sqrt(phot['aper_sum_bkgsub'])
        error_scatter_sky = aperture.area * bkg_stdev**2
        error_mean_sky = bkg_stdev**2 * aperture.area**2 / annulus_aperture.area
    
        fluxerr = np.sqrt(error_poisson**2/epadu + error_scatter_sky + error_mean_sky)
        phot['aperture_sum_err'] = fluxerr
    return phot


def jwst_aperture_phot(fname,ra,dec,
                    filt,ee='r70'):
    try:
        force_ra = float(ra)
        force_dec = float(dec)
        unit = u.deg
    except:
        unit = (u.hourangle, u.deg)
    
    if isinstance(ee,str):
        radius,apcorr,skyan_in,skyan_out = get_apcorr_params(fname,int(ee[1:]))
    else:
        radius,apcorr,skyan_in,skyan_out = ee,1,ee+1,ee+3
    #radius =1.8335238
    #apcorr = aper_func(radius)
    print(filt,radius,apcorr)
    #radius,apcorr = 1.83,1
    image = fits.open(fname)
    
    data = image['SCI',1].data#*image['AREA',1].data
    err = image['ERR',1].data
    imwcs = wcs.WCS(image[1])
    print(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(ra, dec,unit=unit),imwcs))
    #positions = np.atleast_2d(np.flip([582.80256776,819.78997553]))#
    positions = np.atleast_2d(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(ra, dec,unit=unit),imwcs))
    
    imh = image['SCI',1].header
    area = image[1].header['PIXAR_SR']
    aa = np.argwhere(data < 0)
    
    for i in np.arange(0, len(aa), 1):
        data[aa[i][0], aa[i][1]] = 0
    sky = {'sky_in':skyan_in,'sky_out':skyan_out}
    #with datamodels.open(fname) as model:
    #    dat = model.data
    #    err = model.err

    #phot = generic_aperture_phot(data,positions,radius,sky,error=image['ERR',1].data)
    phot = generic_aperture_phot(data,positions,radius,sky,error=err)

    phot['aper_sum_corrected'] = phot['aper_sum_bkgsub'] * apcorr
    phot['aperture_sum_err']*=apcorr
    phot['magerr'] = 2.5 * np.log10(1.0 + (phot['aperture_sum_err']/phot['aper_sum_bkgsub']))

    pixel_scale = wcs.utils.proj_plane_pixel_scales(imwcs)[0]  * imwcs.wcs.cunit[0].to('arcsec')
    flux_units = u.MJy / u.sr * (pixel_scale * u.arcsec)**2
    flux = phot['aper_sum_corrected']*flux_units
    phot['mag'] = flux.to(u.ABmag).value

    return phot


def hst_apcorr(ap,filt,inst):
    if inst=='ir':
        if not os.path.exists('ir_ee_corrections.csv'):
            urllib.request.urlretrieve('https://www.stsci.edu/files/live/sites/www/files/home/hst/'+\
                                       'instrumentation/wfc3/data-analysis/photometric-calibration/'+\
                                       'ir-encircled-energy/_documents/ir_ee_corrections.csv',
                                       'ir_ee_corrections.csv')
        
        ee = Table.read('ir_ee_corrections.csv',format='ascii')
        ee.remove_column('FILTER')
        waves = ee['PIVOT']
        ee.remove_column('PIVOT')
    else:
        if not os.path.exists('wfc3uvis2_aper_007_syn.csv'):
            urllib.request.urlretrieve('https://www.stsci.edu/files/live/sites/www/files/home/hst/'+\
                                   'instrumentation/wfc3/data-analysis/photometric-calibration/'+\
                                   'uvis-encircled-energy/_documents/wfc3uvis2_aper_007_syn.csv',
                                      'wfc3uvis2_aper_007_syn.csv')
        ee = Table.read('wfc3uvis2_aper_007_syn.csv',format='ascii')
    
        ee.remove_column('FILTER')
        waves = ee['WAVELENGTH']
        ee.remove_column('WAVELENGTH')
    ee_arr = np.array([ee[col] for col in ee.colnames])
    apps = [float(x.split('#')[1]) for x in ee.colnames]
    interp = scipy.interpolate.interp2d(waves,apps,ee_arr)
    try:
        filt_wave = sncosmo.get_bandpass(filt).wave_eff
    except:
        filt_wave = sncosmo.get_bandpass('uv'+filt).wave_eff
    return(interp(filt_wave,ap))

def hst_get_zp(filt,zpsys='ab'):
    if zpsys.lower()=='ab':
        return {'F098M':25.666,'F105W':26.264,'F110W':26.819,'F125W':26.232,'F140W':26.450,'F160W':25.936}[filt]
    elif zpsys.lower()=='vega':
        return {'F098M':25.090,'F105W':25.603,'F110W':26.042,'F125W':25.312,'F140W':25.353,'F160W':24.662}[filt]
    else:
        print('unknown zpsys')
        return

def hst_aperture_phot(fname,force_ra,force_dec,filt,radius=3,
                      skyan_in=4,skyan_out=8):
    data_file = fits.open(fname)
    drc_dat = data_file['SCI',1]
    if data_file[1].header['BUNIT']=='ELECTRON':
        epadu = 1
    else:
        epadu = data_file[0].header['EXPTIME']
    try:
        force_ra = float(force_ra)
        force_dec = float(force_dec)
        unit = u.deg
    except:
        unit = (u.hourangle, u.deg)
    sky_location = SkyCoord(force_ra,force_dec,unit=unit)
    imwcs = wcs.WCS(drc_dat.header,data_file)
    x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
    px_scale = wcs.utils.proj_plane_pixel_scales(imwcs)[0] * imwcs.wcs.cunit[0].to('arcsec')
    try:
        zp = hst_get_zp(filt,'ab')
        inst = 'ir'
    except:
        inst = 'uvis'
    phot = generic_aperture_phot(drc_dat.data,np.atleast_2d([x,y]),
                                       radius,{'sky_in':skyan_in,'sky_out':skyan_out},epadu=epadu)
    phot['magerr'] = 1.086 * phot['aperture_sum_err']/phot['aper_sum_bkgsub']
    
    apcorr = hst_get_ee_corr(radius*px_scale,filt,inst)
    if inst=='ir':
        print(apcorr)
        ee_corr = 2.5*np.log10(apcorr)
        zp = hst_get_zp(filt,'ab')
        phot['aper_sum_corrected'] = phot['aper_sum_bkgsub']/apcorr
        phot['mag'] = -2.5*np.log10(phot['aper_sum_corrected'])+zp
    else:
        try:
            hdr = drc_dat.header
            photflam = hdr['PHOTFLAM']
        except:
            hdr = fits.open(data_file)[0].header
            photflam = hdr['PHOTFLAM']
        photplam = drc_dat.header['PHOTPLAM']

        ee_corr = 2.5*np.log10(apcorr)
        zp = -2.5*np.log10(photflam)-5*np.log10(photplam)-2.408
    phot['aper_sum_corrected'] = phot['aper_sum_bkgsub'] / apcorr
    phot['aperture_sum_err']/=apcorr
    phot['mag'] = -2.5*np.log10(phot['aper_sum_corrected']) + zp
    return(phot)

def simple_aperture_sum(data,positions,radius):
    aperture = CircularAperture(positions, r=radius)
    phot = aperture_photometry(data, aperture, method='exact')
    return phot['aperture_sum']