import warnings,os,sys,time,glob
import numpy as np
import urllib.request
import scipy

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
warnings.simplefilter('ignore')
import jwst
from jwst import datamodels
from jwst import source_catalog
from jwst.source_catalog import reference_data

def get_apcorr_params(fname,ee=70):
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

def jwst_get_ap_corr_table():
    
    cnames = ['filter', 'pupil', 'wave', 'r10', 'r20', 'r30', 'r40', 'r50', 'r60', 'r70', 'r80',
                                        'r85', 'r90', 'sky_flux_px', 'apcorr10', 'apcorr20', 'apcorr30', 'apcorr40',
                                        'apcorr50', 'apcorr60', 'apcorr70', 'apcorr80', 'apcorr85', 'apcorr90', 'sky_in',
                                        'sky_out']
    
    if not os.path.isfile('./jwst_aperture_corrections.txt'):
        if os.path.isfile('./aperture_correction_table.txt'):
            ap_tab = './aperture_correction_table.txt'
        else:
            print("Downloading the nircam aperture correction table")

            boxlink_apcorr_table = 'https://data.science.stsci.edu/redirect/JWST/jwst-data_analysis_tools/stellar_photometry/aperture_correction_table.txt'
            boxfile_apcorr_table = './aperture_correction_table.txt'
            urllib.request.urlretrieve(boxlink_apcorr_table, boxfile_apcorr_table)
            ap_tab = './aperture_correction_table.txt'

        aper_table1 = Table.read(ap_tab,format='ascii',names=cnames)
        if os.path.isfile('./jwst_miri_apcorr_0008.fits'):
            ap_tab = Table.read('jwst_miri_apcorr_0008.fits',format='fits')
        else:
            print("Downloading the miri aperture correction table")
            urllib.request.urlretrieve('https://jwst-crds.stsci.edu/unchecked_get/references/jwst/jwst_miri_apcorr_0008.fits',
                                       'jwst_miri_apcorr_0008.fits')
        tab = Table.read('jwst_miri_apcorr_0008.fits',format='fits')
        rows = {k:[] for k in cnames}
        tab=tab[tab['subarray']=='FULL']
        tab.rename_column('subarray','pupil')
        tab.rename_column('skyin','sky_in')
        tab.rename_column('skyout','sky_out')
        for i in range(len(np.unique(tab['filter']))):
            temp_tab = tab[tab['filter']==np.unique(tab['filter'])[i]]
            for c in cnames:
                if c in tab.colnames:
                    rows[c].append(temp_tab[0][c])
                elif c=='wave':
                    rows[c].append(int(temp_tab[0]['filter'][1:-1]))
                elif c=='sky_flux_px':
                    rows[c].append(np.nan)
                elif c.startswith('r'):
                    ind = np.where(temp_tab['eefraction']==float(c[1:])/100)[0]
                    if len(ind)>0:
                        rows[c].append(temp_tab[ind]['radius'])
                        rows['apcorr'+c[1:]].append(temp_tab[ind]['apcorr'])
                    else:
                        rows[c].append(np.nan)
                        rows['apcorr'+c[1:]].append(np.nan)
        
        aper_table2 = Table(rows)
        for k in aper_table1.colnames:
            aper_table1[k] = np.array(aper_table1[k]).flatten()
            aper_table2[k] = np.array(aper_table2[k]).flatten()
        aper_table = vstack([aper_table1,aper_table2])
        
        aper_table.write('./jwst_aperture_corrections.txt',format='ascii')
    else:
        aper_table = Table.read('./jwst_aperture_corrections.txt',format='ascii')
    return(aper_table)

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
    bkg_median = np.array(bkg_median)

    bkg_stdev = np.array(bkg_stdev)

    phot = aperture_photometry(data, aperture, method='exact',error=error)
    phot['annulus_median'] = bkg_median
    phot['aper_bkg'] = bkg_median * aperture.area
    phot['aper_sum_bkgsub'] = phot['aperture_sum'] - phot['aper_bkg']
    if error is None:
        error_poisson = np.sqrt(phot['aper_sum_bkgsub'])

        error_scatter_sky = aperture.area * bkg_stdev**2
        error_mean_sky = bkg_stdev**2 * aperture.area**2 / annulus_aperture.area
        print(float(phot['aper_sum_bkgsub']),error_poisson,error_scatter_sky,error_mean_sky)    
        fluxerr = np.sqrt(error_poisson**2/epadu + error_scatter_sky + error_mean_sky)
        phot['aperture_sum_err'] = fluxerr
    return phot


def jwst_aperture_phot(fname,ra,dec,
                    filt,ee='r70',xy=None):
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
    #err = image['ERR',1].data
    imwcs = wcs.WCS(image[1])
    #print(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(ra, dec,unit=unit),imwcs))
    #positions = np.atleast_2d(np.flip([582.80256776,819.78997553]))#
    if xy is None:
        positions = np.atleast_2d(astropy.wcs.utils.skycoord_to_pixel(SkyCoord(ra, dec,unit=unit),imwcs))
    else:
        positions = np.atleast_2d(xy)
    
    imh = image['SCI',1].header
    area = image[1].header['PIXAR_SR']
    aa = np.argwhere(data < 0)
    
    for i in np.arange(0, len(aa), 1):
        data[aa[i][0], aa[i][1]] = 0
    sky = {'sky_in':skyan_in,'sky_out':skyan_out}
    with datamodels.open(fname) as model:
       dat = model.data
       err = None#model.err
    

    #phot = generic_aperture_phot(data,positions,radius,sky,error=image['ERR',1].data)
    phot = generic_aperture_phot(data,positions,radius,sky,error=err,epadu=image[1].header['PHOTMJSR'])


    phot['aper_sum_corrected'] = phot['aper_sum_bkgsub'] * apcorr
    phot['aperture_sum_err']*=apcorr
    phot['magerr'] = 2.5 * np.log10(1.0 + (phot['aperture_sum_err']/phot['aper_sum_bkgsub']))

    pixel_scale = wcs.utils.proj_plane_pixel_scales(imwcs)[0]  * imwcs.wcs.cunit[0].to('arcsec')
    flux_units = u.MJy / u.sr * (pixel_scale * u.arcsec)**2
    flux = phot['aper_sum_corrected']*flux_units
    phot['mag'] = flux.to(u.ABmag).value

    return phot


def hst_get_ee_corr(ap,filt,inst):
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
    filt_wave = sncosmo.get_bandpass(filt).wave_eff
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



