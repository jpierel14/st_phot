import astropy
import numpy as np
import os,scipy
import webbpsf
import matplotlib.pyplot as plt
from poppy.utils import radial_profile

def hst_get_zp(filt,zpsys='ab'):
    if zpsys.lower()=='ab':
        return {'F098M':25.666,'F105W':26.264,'F110W':26.819,'F125W':26.232,'F140W':26.450,'F160W':25.936}[filt]
    elif zpsys.lower()=='vega':
        return {'F098M':25.090,'F105W':25.603,'F110W':26.042,'F125W':25.312,'F140W':25.353,'F160W':24.662}[filt]
    else:
        print('unknown zpsys')
        return

def calibrate_JWST_flux(flux,fluxerr,imwcs,units = astropy.units.MJy):
    magerr = 2.5 * np.log10(1.0 + (fluxerr/flux))

    pixel_scale = astropy.wcs.utils.proj_plane_pixel_scales(imwcs)[0]  * imwcs.wcs.cunit[0].to('arcsec')
    flux_units = astropy.units.MJy / astropy.units.sr * (pixel_scale * astropy.units.arcsec)**2
    flux = flux*flux_units
    fluxerr = fluxerr*flux_units
    flux = flux.to(units)
    fluxerr = fluxerr.to(units)
    mag = flux.to(astropy.units.ABmag)
    zp = mag.value+2.5*np.log10(flux.value)
    
    return(flux.value,fluxerr.value,mag.value,magerr,zp)

def calibrate_HST_flux(flux,fluxerr,primary_header,sci_header):

    magerr = 1.086 * fluxerr/flux
    instrument = primary_header['INSTRUME']

    if instrument=='IR':
        zp = hst_get_zp(filt,'ab')
    else:
        try:
            photflam = sci_header['PHOTFLAM']
        except:
            photflam = primary_header['PHOTFLAM']
        photplam = sci_header['PHOTPLAM']
        zp = -2.5*np.log10(photflam)-5*np.log10(photplam)-2.408

    mag = -2.5*np.log10(flux)+zp
    return(flux,fluxerr,mag,magerr,zp)


def calc_jwst_psf_corr(ap_rad,instrument,band,imwcs,oversample=4,show_plot=False,psf=None):
    if psf is None:
        inst = webbpsf.instrument(instrument)
        inst.filter = band
        psf = inst.calc_psf(oversample=oversample)

    if show_plot:
        webbpsf.display_ee(psf)
        plt.show()
    radius, profile, ee = radial_profile(psf, ee=True, ext=0)
    ee_func = scipy.interpolate.interp1d(radius,ee)
    pixel_scale = astropy.wcs.utils.proj_plane_pixel_scales(imwcs)[0]  * imwcs.wcs.cunit[0].to('arcsec')
    return(1/ee_func(ap_rad*pixel_scale),psf)

    #print(ee_func(np.max(radius))/ee_func(rad_func(.857)))#/pixel_scale)


