import numpy as np
import astropy
import sncosmo
import nestle
import corner
from collections import OrderedDict
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import os,sys
from stsci.skypac import pamutils

from .util import generic_aperture_phot,jwst_apcorr,hst_apcorr,simple_aperture_sum
from .cal import calibrate_JWST_flux,calibrate_HST_flux,calc_jwst_psf_corr,calc_hst_psf_corr,JWST_mag_to_flux

__all__ = ['observation3','observation']
class observation3():
    """
    st_phot class for level 3 (drizzled) data
    """
    def __init__(self,fname):
        self.fname = fname
        self.fits = astropy.io.fits.open(self.fname)
        self.data = self.fits['SCI',1].data
        try:
            self.err = np.zeros(self.data.shape)
            #self.err = self.fits['ERR',1]
        except:
            self.err = 1./np.sqrt(self.fits['WHT',1])
        self.prim_header = self.fits[0].header
        self.sci_header = self.fits['SCI',1].header
        self.wcs = astropy.wcs.WCS(self.sci_header)
        try:
            self.telescope = self.prim_header['TELESCOP']
            self.instrument = self.prim_header['INSTRUME']
            try:
                self.filter = self.prim_header['FILTER']
            except:
                if 'CLEAR' in self.prim_header['FILTER1']:
                    self.filter = self.prim_header['FILTER2']
                else:
                    self.filter = self.prim_header['FILTER1']
        except:
            self.telescope = 'JWST'
            self.instrument = 'NIRCam'


    def aperture_photometry(self,sky_location,xy_positions=None,
                radius=None,encircled_energy=None,skyan_in=None,skyan_out=None):
        """
        We do not provide a PSF photometry method for level 3 data, but aperture photometry is possible.

        Parameters
        ----------
        sky_location : :class:`~astropy.coordinates.SkyCoord`
            The location of your source
        xy_positions : list
            The xy position of your source (will be used in place of sky_location)
        radius : float
            Needed for HST observations to define the aperture radius
        encircled_energy: int
            Multiple of 10 to define the JWST aperture information
        skyan_in : float
            For HST, defines the inner radius of the background sky annulus
        skyan_out : float
            For HST, defines the outer radius of the background sky annulus

        Returns
        -------
        aperture_result
        """
        assert radius is not None or encircled_energy is not None, 'Must supply radius or ee'
        assert (self.telescope.lower()=='hst' and radius is not None) or \
        (self.telescope.lower()=='jwst' and encircled_energy is not None),\
        'Must supply radius for hst or ee for jwst'

        result = {'pos_x':[],'pos_y':[],'aper_bkg':[],'aperture_sum':[],'aperture_sum_err':[],
                  'aper_sum_corrected':[],'aper_sum_bkgsub':[],'annulus_median':[],'exp':[]}
        result_cal = {'flux':[],'flux_err':[],'filter':[],'zp':[],'mag':[],'magerr':[],'zpsys':[],'exp':[],'mjd':[]}
        
        if xy_positions is None:
            positions = np.atleast_2d(astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs))
        else:
            positions = np.atleast_2d(xy_positions)
        if self.telescope=='JWST':
            radius,apcorr,skyan_in,skyan_out = jwst_apcorr(self.fname,encircled_energy)
            epadu = self.sci_header['XPOSURE']*self.sci_header['PHOTMJSR']
        else:
            if self.sci_header['BUNIT']=='ELECTRON':
                epadu = 1
            else:
                epadu = self.prim_header['EXPTIME']
            px_scale = astropy.wcs.utils.proj_plane_pixel_scales(self.wcs)[0] *\
                                                                         self.wcs.wcs.cunit[0].to('arcsec')
            apcorr = hst_apcorr(radius*px_scale,self.filter,self.instrument)
            if skyan_in is None:
                skyan_in = radius*3
            if skyan_out is None:
                skyan_out = radius*4

        sky = {'sky_in':skyan_in,'sky_out':skyan_out}
        phot = generic_aperture_phot(self.data,positions,radius,sky,#error=self.err,
                                            epadu=epadu)
        for k in phot.keys():
            if k in result.keys():
                result[k].append(float(phot[k]))
        result['pos_x'].append(positions[0][0])
        result['pos_y'].append(positions[0][1])
        if self.telescope.lower()=='jwst':
            result['aper_sum_corrected'].append(float(phot['aper_sum_bkgsub'] * apcorr))
            result['aperture_sum_err'][-1]*= apcorr
        else:
            result['aper_sum_corrected'].append(float(phot['aper_sum_bkgsub'] / apcorr))
            result['aperture_sum_err'][-1]/= apcorr
        result['exp'].append(os.path.basename(self.fname))

        if self.telescope=='JWST':
            flux,fluxerr,mag,magerr,zp = calibrate_JWST_flux(result['aper_sum_corrected'][-1],
                                                          result['aperture_sum_err'][-1],
                                                          self.wcs)
            mjd = self.prim_header['MJD-AVG']
        else:
            flux,fluxerr,mag,magerr,zp = calibrate_HST_flux(result['aper_sum_corrected'][-1],
                                                          result['aperture_sum_err'][-1],
                                                          self.prim_header,
                                                          self.sci_header)
            mjd = np.median([self.prim_header['EXPSTART'],self.prim_header['EXPEND']])
        result_cal['flux'].append(flux)
        result_cal['flux_err'].append(fluxerr)
        result_cal['mag'].append(mag)
        result_cal['magerr'].append(magerr)
        result_cal['filter'].append(self.filter)
        result_cal['zp'].append(zp)
        result_cal['zpsys'].append('ab')
        result_cal['exp'].append(os.path.basename(self.fname))
        result_cal['mjd'].append(mjd)
        res = sncosmo.utils.Result(radius=radius,
                   apcorr=apcorr,
                   sky_an=sky,
                   phot_table=astropy.table.Table(result),
                   phot_cal_table=astropy.table.Table(result_cal)
                   )
        self.aperture_result = res
        return self.aperture_result


class observation():
    """
    st_phot class for level 2 (individual exposures, cals, flts) data
    """
    def __init__(self,exposure_fnames,sci_ext=1):
        self.exposure_fnames = exposure_fnames if not isinstance(exposure_fnames,str) else [exposure_fnames]
        self.exposures = [astropy.io.fits.open(f) for f in self.exposure_fnames]
        self.data_arr = [im['SCI',sci_ext].data for im in self.exposures]
        self.err_arr = [im['ERR',sci_ext].data for im in self.exposures]
        self.prim_headers = [im[0].header for im in self.exposures]
        self.sci_headers = [im['SCI',sci_ext].header for im in self.exposures]
        self.wcs_list = [astropy.wcs.WCS(hdr,dat) for hdr,dat in zip(self.sci_headers,self.exposures)]
        self.n_exposures = len(self.exposures)
        self.telescope = self.prim_headers[0]['TELESCOP']
        self.instrument = self.prim_headers[0]['INSTRUME']
        try:
            self.detector = self.prim_headers[0]['DETECTOR']
        except:
            self.detector = None

        if self.sci_headers[0]['BUNIT'] in ['ELECTRON','ELECTRONS']:
            self.data_arr = [self.data_arr[i]/self.prim_headers[i]['EXPTIME'] for i in range(self.n_exposures)]
            self.err_arr = [self.err_arr[i]/self.prim_headers[i]['EXPTIME'] for i in range(self.n_exposures)]

        if 'FILTER' in self.prim_headers[0].keys():
            self.filter = np.unique([hdr['FILTER'] for hdr in self.prim_headers])
        elif 'FILTER1' in self.prim_headers[0].keys():
            if 'CLEAR' in self.prim_headers[0]['FILTER1']:
                self.filter = np.unique([hdr['FILTER2'] for hdr in self.prim_headers])
            else:
                self.filter = np.unique([hdr['FILTER1'] for hdr in self.prim_headers])
        else:
            self.filter = np.unique([hdr['FILTER'] for hdr in self.sci_headers])
        if len(self.filter)>1:
            raise RuntimeError("Each observation should only have one filter.")
        self.filter = self.filter[0]
        if self.telescope.lower()=='jwst':
            self.pams = [im['AREA'].data for im in self.exposures]
        else:
            self.pams = []
            for fname in self.exposure_fnames:
                 pamutils.pam_from_file(fname, ('sci', sci_ext), 'temp_pam.fits')
                 self.pams.append(astropy.io.fits.open('temp_pam.fits')[0].data)
            os.remove('temp_pam.fits')
        
        self.data_arr_pam = [im*pam for im,pam in zip(self.data_arr,self.pams)]

    def plant_psf(self,psf_model,plant_locations,magnitudes):
        """
        PSF planting class. Output files will be the same directory
        as the data files, but with _plant.fits added the end. 

        Parameters
        ----------
        psf_model : :class:`~photutils.psf.EPSFModel`
            In reality this does not need to be an EPSFModel, but just any
            photutils psf model class.
        plant_locations : list
            The location(s) to plant the psf
        magnitudes:
            The magnitudes to plant your psf (matching length of plant_locations)
        """

        if not isinstance(plant_locations,(list,np.ndarray)):
            plant_locations = [plant_locations]
        if isinstance(magnitudes,(int,float)):
            magnitudes = [magnitudes]*len(plant_locations)
        if not isinstance(psf_model,list):
            psf_model = [psf_model]
        assert len(psf_model)==len(plant_locations)==len(magnitudes), "Must supply same number of psfs,plant_locations,mags"
        #psf_corr,mod_psf = calc_jwst_psf_corr(psf_model.data.shape[0]/2,self.instrument,
        #    self.filter,self.wcs_list[0])
        for i in range(self.n_exposures):
            temp = astropy.io.fits.open(self.exposure_fnames[i])
            for j in range(len(plant_locations)):
                if isinstance(plant_locations[i],astropy.coordinates.SkyCoord):
                    y,x = astropy.wcs.utils.skycoord_to_pixel(plant_locations[i],self.wcs_list[i])
                else:
                    x,y = plant_locations[i]
                flux = JWST_mag_to_flux(magnitudes[j],self.wcs_list[i])
                psf_model[j].x_0 = x
                psf_model[j].y_0 = y
                psf_model[j].flux = flux/np.sum(psf_model[j].data)#/psf_corr
                #psf_arr = flux*psf_model.data/astropy.nddata.extract_array(\
                #    self.pams[i],psf_model.data.shape,[x,y])
                yf, xf = np.mgrid[0:temp['SCI',1].data.shape[0],0:temp['SCI',1].data.shape[1]].astype(int)
                psf_arr = psf_model[j](yf,xf)
            

                temp['SCI',1].data+=psf_arr# = astropy.nddata.add_array(temp['SCI',1].data,
                    #psf_arr,[x,y])
            temp.writeto(self.exposure_fnames[i].replace('.fits','_plant.fits'),overwrite=True)


    


    def aperture_photometry(self,sky_location,xy_positions=None,
                radius=None,encircled_energy=None,skyan_in=None,skyan_out=None):
        """
        Aperture photometry class for level 2 data

        Parameters
        ----------
        sky_location : :class:`~astropy.coordinates.SkyCoord`
            The location of your source
        xy_positions : list
            The xy position of your source (will be used in place of sky_location)
        radius : float
            Needed for HST observations to define the aperture radius
        encircled_energy: int
            Multiple of 10 to define the JWST aperture information
        skyan_in : float
            For HST, defines the inner radius of the background sky annulus
        skyan_out : float
            For HST, defines the outer radius of the background sky annulus

        Returns
        -------
        aperture_result
        """
        assert radius is not None or encircled_energy is not None, 'Must supply radius or ee'
        assert (self.telescope.lower()=='hst' and radius is not None) or \
        (self.telescope.lower()=='jwst' and encircled_energy is not None),\
        'Must supply radius for hst or ee for jwst'

        result = {'pos_x':[],'pos_y':[],'aper_bkg':[],'aperture_sum':[],'aperture_sum_err':[],
                  'aper_sum_corrected':[],'aper_sum_bkgsub':[],'annulus_median':[],'exp':[]}
        result_cal = {'flux_cal':[],'flux_cal_err':[],'filter':[],'zp':[],'mag':[],'magerr':[],'zpsys':[],'exp':[]}
        for i in range(self.n_exposures):
            if xy_positions is None:
                positions = np.atleast_2d(astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[i]))
            else:
                positions = np.atleast_2d(xy_positions)
            if self.telescope=='JWST':
                radius,apcorr,skyan_in,skyan_out = jwst_apcorr(self.exposure_fnames[i],encircled_energy)
                epadu = self.sci_headers[i]['XPOSURE']*self.sci_headers[i]['PHOTMJSR']
            else:
                if self.sci_headers[i]['BUNIT']=='ELECTRON':
                    epadu = 1
                else:
                    epadu = self.prim_headers[i]['EXPTIME']
                px_scale = astropy.wcs.utils.proj_plane_pixel_scales(self.wcs_list[0])[0] *\
                                                                             self.wcs_list[0].wcs.cunit[0].to('arcsec')
                apcorr = hst_apcorr(radius*px_scale,self.filter,self.instrument)
                if skyan_in is None:
                    skyan_in = radius*3
                if skyan_out is None:
                    skyan_out = radius*4

            sky = {'sky_in':skyan_in,'sky_out':skyan_out}
            phot = generic_aperture_phot(self.data_arr_pam[i],positions,radius,sky,error=self.err_arr[i],
                                                epadu=epadu)
            for k in phot.keys():
                if k in result.keys():
                    result[k].append(float(phot[k]))
            result['pos_x'].append(positions[0][0])
            result['pos_y'].append(positions[0][1])
            if self.telescope.lower()=='jwst':
                result['aper_sum_corrected'].append(float(phot['aper_sum_bkgsub'] * apcorr))
                result['aperture_sum_err'][-1]*= apcorr
            else:
                result['aper_sum_corrected'].append(float(phot['aper_sum_bkgsub'] / apcorr))
                result['aperture_sum_err'][-1]/= apcorr
            result['exp'].append(os.path.basename(self.exposure_fnames[i]))

            if self.telescope=='JWST':
                flux,fluxerr,mag,magerr,zp = calibrate_JWST_flux(result['aper_sum_corrected'][-1],
                                                              result['aperture_sum_err'][-1],
                                                              self.wcs_list[i])
            else:
                flux,fluxerr,mag,magerr,zp = calibrate_HST_flux(result['aper_sum_corrected'][-1],
                                                              result['aperture_sum_err'][-1],
                                                              self.prim_headers[i],
                                                              self.sci_headers[i])
            result_cal['flux_cal'].append(flux)
            result_cal['flux_cal_err'].append(fluxerr)
            result_cal['mag'].append(mag)
            result_cal['magerr'].append(magerr)
            result_cal['filter'].append(self.filter)
            result_cal['zp'].append(zp)
            result_cal['zpsys'].append('ab')
            result_cal['exp'].append(os.path.basename(self.exposure_fnames[i]))

        res = sncosmo.utils.Result(radius=radius,
                   apcorr=apcorr,
                   sky_an=sky,
                   phot_table=astropy.table.Table(result),
                   phot_cal_table=astropy.table.Table(result_cal)
                   )
        self.aperture_result = res

    def psf_photometry(self,psf_model,sky_location=None,xy_positions=[],fit_width=None,background=None,
                        fit_flux='single',fit_centroid='pixel',fit_bkg=False,bounds={},npoints=100,use_MLE=False,
                        maxiter=None):
        """
        st_phot psf photometry class for level 2 data.

        Parameters
        ----------
        psf_model : :class:`~photutils.psf.EPSFModel`
            In reality this does not need to be an EPSFModel, but just any
            photutils psf model class.
        sky_location : :class:`~astropy.coordinates.SkyCoord`
            Location of your source
        xy_positions : list
            xy position of your source in each exposure. Must supply this or
            sky_location but this takes precedent.
        fit_width : int
            PSF width to fit (recommend odd number)
        background : float or list
            float, list of float, array, or list of array defining the background
            of your data. If you define an array, it should be of the same shape
            as fit_width
        fit_flux : str
            One of 'single','multi','fixed'. Single is a single flux across all
            exposures, multi fits a flux for every exposure, and fixed only
            fits the position
        fit_centroid : str
            One of 'pixel','wcs','fixed'. Pixel fits a pixel location of the
            source in each exposure, wcs fits a single RA/DEC across all exposures,
            and fixed only fits the flux and not position.
        fit_bkg : bool
            Fit for a constant background simultaneously with the PSF fit.
        bounds : dict
            Bounds on each parameter. 
        npoints : int
            Number of points in the nested sampling (higher is better posterior sampling, 
            but slower)
        use_MLE : bool
            Use the maximum likelihood to define best fit parameters, otherwise use a weighted
            average of the posterior samples
        maxiter : None or int
            If None continue sampling until convergence, otherwise defines the max number of iterations
        """
        assert sky_location is not None or len(xy_positions)==self.n_exposures,\
        "Must supply sky_location or xy_positions for every exposure"

        assert fit_flux in ['single','multi','fixed'],\
                "fit_flux must be one of: 'single','multi','fixed'"

        assert fit_centroid in ['pixel','wcs','fixed'],\
            "fit_centroid must be one of: 'pixel','wcs','fixed'"

        assert len(bounds)>0,\
            "Must supply bounds"

        if fit_flux=='fixed' and fit_centroid=='fixed':
            print('Nothing to do, fit flux and/or position.')
            return


        if fit_width is None:
            try:
                fit_width = psf_model.data.shape[0]
            except:
                 RuntimeError("If you do not supply fit_width, your psf needs to have a data attribute (i.e., be an ePSF")

        if fit_width%2==0:
            print('PSF fitting width is even, subtracting 1.')
            fit_width-=1

        centers = []
        all_xf = []
        all_yf = []
        cutouts = []
        cutout_errs = []
        fluxg = []

        if background is None:
            all_bg_est = [0]*self.n_exposures #replace with bkg method
            if not fit_bkg:
                print('Warning: No background subtracting happening here.')
        elif isinstance(background,(int,float)):
            all_bg_est = [background]*self.n_exposures
        else:
            all_bg_est = background

        if not isinstance(psf_model,list):
            self.psf_model_list = []
            for i in range(self.n_exposures):
                self.psf_model_list.append(deepcopy(psf_model))
        else:
            self.psf_model_list = psf_model

        for im in range(self.n_exposures):
            if len(xy_positions)==self.n_exposures:
                xi,yi = xy_positions[im]
            else:
                yi,xi = astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[im])

            centers.append([xi,yi])
            yg, xg = np.mgrid[-1*(fit_width-1)/2:(fit_width+1)/2,
                              -1*(fit_width-1)/2:(fit_width+1)/2].astype(int)
            yf, xf = yg+np.round(yi).astype(int), xg+np.round(xi).astype(int)
            
            all_xf.append(xf)
            all_yf.append(yf)
            cutout = self.data_arr_pam[im][xf, yf]
            
            err = self.err_arr[im][xf, yf]
            err[err<=0] = np.max(err)
            cutouts.append(cutout-all_bg_est[im])
            cutout_errs.append(err)
            
            if fit_flux!='fixed':
                f_guess = np.sum(cutout-all_bg_est[im])
                fluxg.append(f_guess)

        if fit_flux=='single':
            fluxg = [np.median(fluxg)]
            pnames = ['flux']
        elif fit_flux=='multi':
            pnames = ['flux%i'%i for i in range(self.n_exposures)]
        else:
            pnames = []
        
        if fit_centroid!='fixed':
            if fit_centroid=='pixel':
                for i in range(self.n_exposures):
                    pnames.append(f'x{i}')
                    pnames.append(f'y{i}')
            else:
                pnames.append(f'ra')
                pnames.append(f'dec')
        pnames = np.array(pnames).ravel()
        if fit_centroid=='wcs':
            new_centers = []
            n = 0
            for center in centers:     
                sc = astropy.wcs.utils.pixel_to_skycoord(center[1],center[0],self.wcs_list[n])
                new_centers.append([sc.ra.value,sc.dec.value])
                n+=1

            p0s = np.append(fluxg,[np.median(new_centers,axis=0)]).flatten()        
        elif fit_centroid=='pixel':
            p0s = np.append(fluxg,centers).flatten()        
        else:
            p0s = np.array(fluxg)
            for i in range(self.n_exposures):
                self.psf_model_list[i].x_0 = centers[i][0]
                self.psf_model_list[i].y_0 = centers[i][1]


        if not np.all([x in bounds.keys() for x in pnames]):
            pbounds = {}
            for i in range(len(pnames)):
                if 'flux' in pnames[i]:
                    pbounds[pnames[i]] = np.array(bounds['flux'])+p0s[i]
                    if pbounds[pnames[i]][0]<0:
                        pbounds[pnames[i]][0] = 0
                    if pbounds[pnames[i]][1]<=0:
                        raise RuntimeError('Your flux bounds are both <=0.')
                else:
                    if fit_centroid=='wcs':
                        px_scale = astropy.wcs.utils.proj_plane_pixel_scales(self.wcs_list[0])[0] *\
                                                                             self.wcs_list[0].wcs.cunit[0].to('deg')

                        pbounds[pnames[i]] = np.array(bounds['centroid'])*px_scale+p0s[i]
                        #pbounds[pnames[i]] = temp_bounds+p0s[i]
                        # todo check inside wcs
                    else:
                        pbounds[pnames[i]] = np.array(bounds['centroid'])+p0s[i]
                        if pbounds[pnames[i]][0]<0:
                            pbounds[pnames[i]][0] = 0
                        
        else:
            pbounds = bounds    

        if fit_bkg:
            assert 'bkg' in bounds.keys(),"Must supply bounds for bkg"

            if fit_flux=='multi':
                to_add = []
                for i in range(len(pnames)):
                    if 'flux' in pnames[i]:
                        to_add.append('bkg%s'%pnames[i][-1])
                        pbounds['bkg%s'%pnames[i][-1]] = bounds['bkg']
                pnames = np.append(pnames,to_add)
                
            else:
                pnames = np.append(pnames,['bkg'])
                pbounds['bkg'] = bounds['bkg']


        self.nest_psf(pnames,pbounds,cutouts,cutout_errs,all_xf,all_yf,
                        psf_width=fit_width,npoints=npoints,use_MLE=use_MLE,maxiter=maxiter)

    
        result_cal = {'ra':[],'ra_err':[],'dec':[],'dec_err':[],'x':[],'x_err':[],
                      'y':[],'y_err':[],'mjd':[],
                      'flux':[],'fluxerr':[],'filter':[],
                      'zp':[],'mag':[],'magerr':[],'zpsys':[],'exp':[]}
        model_psf = None
        psf_pams = []
        for i in range(self.n_exposures):
            if fit_flux=='single':
                flux_var = 'flux'
            else:
                flux_var = 'flux%i'%i 

            if fit_centroid=='wcs':
                ra = self.psf_result.best[self.psf_result.vparam_names.index('ra')]
                dec = self.psf_result.best[self.psf_result.vparam_names.index('dec')]
                ra_err = self.psf_result.errors['ra']
                dec_err = self.psf_result.errors['dec']
                sky_location = astropy.coordinates.SkyCoord(ra,
                                                            dec,
                                                            unit=astropy.units.deg)
                y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[i])
                raise RuntimeError('Need to implement xy errors from wcs')
            elif fit_centroid=='pixel':
                x = self.psf_result.best[self.psf_result.vparam_names.index('x%i'%i)]
                y = self.psf_result.best[self.psf_result.vparam_names.index('y%i'%i)]
                xerr = self.psf_result.errors['x%i'%i]
                yerr = self.psf_result.errors['y%i'%i]
                sc = astropy.wcs.utils.pixel_to_skycoord(y,x,self.wcs_list[i])
                ra = sc.ra.value
                dec = sc.dec.value
                sc2 = astropy.wcs.utils.pixel_to_skycoord(y+yerr,x+xerr,self.wcs_list[i])
                raerr = np.abs(sc2.ra.value-ra)
                decerr = np.abs(sc2.dec.value-dec)
            else:
                x = self.psf_model_list[i].x_0
                y = self.psf_model_list[i].y_0
                xerr = self.psf_result.errors['x%i'%i]
                yerr = self.psf_result.errors['y%i'%i]
                sc = astropy.wcs.utils.pixel_to_skycoord(y,x,self.wcs_list[i])
                ra = sc.ra.value
                dec = sc.dec.value
                sc2 = astropy.wcs.utils.pixel_to_skycoord(y+yerr,x+xerr,self.wcs_list[i])
                raerr = np.abs(sc2.ra.value-ra)
                decerr = np.abs(sc2.dec.value-dec)

            if 'bkg' in self.psf_result.vparam_names:
                bk_std = self.psf_result.errors['bkg']
            elif 'bkg%i'%i in self.psf_result.vparam_names:
                bk_std = self.psf_result.errors['bkg%i'%i]
            else:
                bk_std = 0
            #psf_pam = astropy.nddata.utils.extract_array(self.pams[i],self.psf_model_list[i].data.shape,[x,y],mode='strict')
            #psf_pams.append(psf_pam)
            
            yf, xf = np.mgrid[0:self.data_arr[i].shape[0],0:self.data_arr[i].shape[1]].astype(int)

            psf_arr = self.psf_model_list[i](yf,xf)#*self.pams[i]#self.psf_pams[i]

            flux_sum = np.sum(psf_arr)#simple_aperture_sum(psf_arr,np.atleast_2d([y,x]),10)

            if self.telescope == 'JWST':
                psf_corr,model_psf = calc_jwst_psf_corr(self.psf_model_list[i].shape[0]/2,self.instrument,self.filter,self.wcs_list[i],psf=model_psf)
                flux,fluxerr,mag,magerr,zp = calibrate_JWST_flux(flux_sum,#*psf_corr,
                    np.sqrt(((self.psf_result.errors[flux_var]/self.psf_result.best[self.psf_result.vparam_names.index(flux_var)])*\
                    flux_sum*psf_corr)**2+bk_std**2),self.wcs_list[i])
            else:
                psf_corr = calc_hst_psf_corr(self.psf_model_list[i].shape[0]/2,self.detector,self.filter,[x,y],'/Users/jpierel/DataBase/HST/psfs')

                flux,fluxerr,mag,magerr,zp = calibrate_HST_flux(flux_sum*psf_corr,
                    np.sqrt(((self.psf_result.errors[flux_var]/self.psf_result.best[self.psf_result.vparam_names.index(flux_var)])*\
                    flux_sum*psf_corr)**2+bk_std**2),self.prim_headers[i],self.sci_headers[i])

            result_cal['x'].append(x)
            result_cal['y'].append(y)
            try:
                result_cal['mjd'].append(self.prim_headers[i]['MJD-AVG'])
            except:
                result_cal['mjd'].append(self.prim_headers[i]['EXPSTART'])
                
            result_cal['flux'].append(flux)
            result_cal['fluxerr'].append(fluxerr)
            result_cal['filter'].append(self.filter)
            result_cal['zp'].append(zp)
            result_cal['mag'].append(mag)
            result_cal['magerr'].append(magerr)
            result_cal['zpsys'].append('ab')
            result_cal['exp'].append(os.path.basename(self.exposure_fnames[i]))
            result_cal['ra_err'].append(raerr)
            result_cal['dec_err'].append(decerr)
            result_cal['ra'].append(ra)
            result_cal['dec'].append(dec)
            result_cal['x_err'].append(xerr)
            result_cal['y_err'].append(yerr)

        self.psf_result.phot_cal_table = astropy.table.Table(result_cal)
        self.psf_pams = psf_pams

        print('Finished PSF psf_photometry with median residuals of %.2f'%\
            (100*np.median([self.psf_result.resid_arr[i]/self.psf_result.data_arr[i] for i in range(self.n_exposures)]))+'%')

    def create_psf_subtracted(self,sci_ext=1,fname=None):
        """
        Use the best fit PSF models to create a PSF-subtracted image

        Parameters
        ----------
        sci_ext : int
            The SCI extension to use (e.g., if this is UVIS)
        fname : str
            Output filename
        """

        try:
            temp = self.psf_result
        except:
            print('Must run PSF fit.')
            return

        for i in range(self.n_exposures):
            x = float(self.psf_model_list[i].x_0.value)
            y = float(self.psf_model_list[i].y_0.value)
            psf_width = self.psf_model_list[i].data.shape[0]
            yf, xf = np.mgrid[0:self.data_arr[i].shape[0],0:self.data_arr[i].shape[1]].astype(int)


            psf_arr = self.psf_model_list[i](yf,xf)*self.pams[i]
            if fname is None:
                temp = astropy.io.fits.open(self.exposure_fnames[i])
                temp['SCI',sci_ext].data = self.data_arr_pam[i]-psf_arr
            else:
                if isinstance(fname,str):
                    temp = astropy.io.fits.open(fname)
                else:
                    temp = astropy.io.fits.open(fname[i])
                temp['SCI',sci_ext].data-= psf_arr 
  

            temp.writeto(self.exposure_fnames[i].replace('.fits','_resid.fits'),overwrite=True)
            temp.close()
        return [self.exposure_fnames[i].replace('.fits','_resid.fits') for i in range(self.n_exposures)]


    def nest_psf(self,vparam_names, bounds,fluxes, fluxerrs,xs,ys,psf_width=7,use_MLE=False,
                       minsnr=0., priors=None, ppfs=None, npoints=100, method='single',
                       maxiter=None, maxcall=None, modelcov=False, rstate=None,
                       verbose=False, warn=True, **kwargs):

        # Taken from SNCosmo nest_lc
        # experimental parameters
        tied = kwargs.get("tied", None)

        

        vparam_names = list(vparam_names)
        if ppfs is None:
            ppfs = {}
        if tied is None:
            tied = {}
        
        # Convert bounds/priors combinations into ppfs
        if bounds is not None:
            for key, val in bounds.items():
                if key in ppfs:
                    continue  # ppfs take priority over bounds/priors
                a, b = val
                if priors is not None and key in priors:
                    # solve ppf at discrete points and return interpolating
                    # function
                    x_samples = np.linspace(0., 1., 101)
                    ppf_samples = sncosmo.utils.ppf(priors[key], x_samples, a, b)
                    f = sncosmo.utils.Interp1D(0., 1., ppf_samples)
                else:
                    f = sncosmo.utils.Interp1D(0., 1., np.array([a, b]))
                ppfs[key] = f

        # NOTE: It is important that iparam_names is in the same order
        # every time, otherwise results will not be reproducible, even
        # with same random seed.  This is because iparam_names[i] is
        # matched to u[i] below and u will be in a reproducible order,
        # so iparam_names must also be.

        iparam_names = [key for key in vparam_names if key in ppfs]

        ppflist = [ppfs[key] for key in iparam_names]
        npdim = len(iparam_names)  # length of u
        ndim = len(vparam_names)  # length of v

        # Check that all param_names either have a direct prior or are tied.
        for name in vparam_names:
            if name in iparam_names:
                continue
            if name in tied:
                continue
            raise ValueError("Must supply ppf or bounds or tied for parameter '{}'"
                             .format(name))

        def prior_transform(u):
            d = {}
            for i in range(npdim):
                d[iparam_names[i]] = ppflist[i](u[i])
            v = np.empty(ndim, dtype=np.float)
            for i in range(ndim):
                key = vparam_names[i]
                if key in d:
                    v[i] = d[key]
                else:
                    v[i] = tied[key](d)
            return v
        
        pos_start = [i for i in range(len(vparam_names))]
        
        if len([x for x in vparam_names if 'flux' in x])>1:
            multi_flux = True
        else:
            multi_flux = False

        if np.any(['dec' in x for x in vparam_names]):
            fit_radec = True
        else:
            fit_radec = False
        if np.any(['y' in x for x in vparam_names]):
            fit_pixel = True
        else:
            fit_pixel = False

        if np.any(['bkg' in x for x in vparam_names]):
            fit_bkg = True
        else:
            fit_bkg = False

        import matplotlib.pyplot as plt
        
        sums = [np.sum(f) for f in fluxes]

        def chisq_likelihood(parameters):
            total = 0
            for i in range(len(fluxes)):
                posx = xs[i]
                posy = ys[i]
                
                if multi_flux:
                    self.psf_model_list[i].flux = parameters[vparam_names.index('flux%i'%i)]
                else:
                    self.psf_model_list[i].flux = parameters[vparam_names.index('flux')]

                if fit_radec:
                    sky_location = astropy.coordinates.SkyCoord(parameters[vparam_names.index('ra')],
                                                                parameters[vparam_names.index('dec')],
                                                                unit=astropy.units.deg)
                    y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[i])
                    self.psf_model_list[i].x_0 = x
                    self.psf_model_list[i].y_0 = y
                elif fit_pixel:
                    self.psf_model_list[i].x_0 = parameters[vparam_names.index('x%i'%(i))]
                    self.psf_model_list[i].y_0 = parameters[vparam_names.index('y%i'%(i))]

                #print(posx)
                #print(posy)
                #print(self.psf_model_list[i].flux)
                #print(fluxes[i])
                mflux = self.psf_model_list[i](posx,posy)
                #print(mflux)
                weights = mflux/np.max(mflux)
                if fit_bkg:
                    if multi_flux:
                        mflux+=parameters[vparam_names.index('bkg%i'%i)]
                    else:
                        mflux+=parameters[vparam_names.index('bkg')]
                #mflux*=self.pams[i][posx,posy]
                # plt.imshow(fluxes[i])
                # plt.show()
                # plt.imshow(mflux)
                # plt.show()
                # print(np.sum(fluxes[i]),np.sum(mflux))
                #sys.exit()
                total+=np.sum(((fluxes[i]-mflux)/fluxerrs[i])**2)#*weights)**2)
                
            return total
        
        
        def loglike(parameters):
            chisq = chisq_likelihood(parameters)
            return(-.5*chisq)
        

        res = nestle.sample(loglike, prior_transform, ndim, npdim=npdim,
                            npoints=npoints, method=method, maxiter=maxiter,
                            maxcall=maxcall, rstate=rstate,
                            callback=(nestle.print_progress if verbose else None))

        vparameters, cov = nestle.mean_and_cov(res.samples, res.weights)

        res = sncosmo.utils.Result(niter=res.niter,
                                   ncall=res.ncall,
                                   logz=res.logz,
                                   logzerr=res.logzerr,
                                   h=res.h,
                                   samples=res.samples,
                                   weights=res.weights,
                                   logvol=res.logvol,
                                   logl=res.logl,
                                   errors=OrderedDict(zip(vparam_names,
                                                          np.sqrt(np.diagonal(cov)))),
                                   vparam_names=copy(vparam_names),
                                   bounds=bounds,
                                   best=vparameters,
                                   data_arr = fluxes,
                                   psf_arr = None,
                                   big_psf_arr = None,
                                   resid_arr = None,
                                   phot_cal_table = None)

        if use_MLE:
            best_ind = res.logl.argmax()
            for i in range(len(vparam_names)):
                res.best[i] = res.samples[best_ind,i]
            params = [[res.samples[best_ind, i]-res.errors[vparam_names[i]], res.samples[best_ind, i], res.samples[best_ind, i]+res.errors[vparam_names[i]]]
                      for i in range(len(vparam_names))]

        all_mflux_arr = []
        all_resid_arr = []
        
        for i in range(len(fluxes)):
            posx = xs[i]
            posy = ys[i]
            
            if multi_flux:
                self.psf_model_list[i].flux = res.best[vparam_names.index('flux%i'%i)]
            else:
                self.psf_model_list[i].flux = res.best[vparam_names.index('flux')]

            if fit_radec:
                sky_location = astropy.coordinates.SkyCoord(res.best[vparam_names.index('ra')],
                                                            res.best[vparam_names.index('dec')],
                                                            unit=astropy.units.deg)
                y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[i])
                self.psf_model_list[i].x_0 = x
                self.psf_model_list[i].y_0 = y
            elif fit_pixel:
                self.psf_model_list[i].x_0 = res.best[vparam_names.index('x%i'%(i))]
                self.psf_model_list[i].y_0 = res.best[vparam_names.index('y%i'%(i))]
            
            mflux = self.psf_model_list[i](posx,posy)
            all_mflux_arr.append(mflux*self.pams[i][posx,posy])

            if fit_bkg:
                if multi_flux:
                    res.data_arr[i]-=res.best[vparam_names.index('bkg%i'%i)]
                else:
                    res.data_arr[i]-=res.best[vparam_names.index('bkg')]
            #mflux*=self.pams[i][posx,posy]
            resid = res.data_arr[i]-mflux
            all_resid_arr.append(resid)
            
        res.psf_arr = all_mflux_arr
        res.resid_arr = all_resid_arr
        self.psf_result = res
        return 

    def plot_psf_fit(self):
        """
        Plot the best-fit PSF model and residuals
        """

        try:
            temp = self.psf_result.data_arr[0]
        except:
            print('Must fit PSF before plotting.')
            return


        fig,axes = plt.subplots(self.n_exposures,3)
        axes = np.atleast_2d(axes)
        for i in range(self.n_exposures):
            norm1 = astropy.visualization.simple_norm(self.psf_result.data_arr[i],stretch='linear')
            axes[i][0].imshow(self.psf_result.data_arr[i],
                norm=norm1)
            axes[i][0].set_title('Data')
            axes[i][1].imshow(self.psf_result.psf_arr[i],
                norm=norm1)
            axes[i][1].set_title('Model')
            axes[i][2].imshow(self.psf_result.resid_arr[i],
                norm=norm1)
            axes[i][2].set_title('Residual')
            for j in range(3):
                axes[i][j].tick_params(
                    axis='both',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    left=False,         # ticks along the top edge are off
                    labelbottom=False,
                    labelleft=False) # labels along the bottom edge are off
        plt.tight_layout()
        plt.show()
        return fig

    def plot_psf_posterior(self,minweight=-np.inf):
        """
        Plot the posterior corner plot from nested sampling

        Parameters
        ----------
        minweight : float
            A minimum weight to show from the nested sampling
            (to zoom in on posterior)
        """
        import corner
        try:
            samples = self.psf_result.samples
        except:
            print('Must fit PSF before plotting.')
            return
        weights = self.psf_result.weights
        samples = samples[weights>minweight]
        weights = weights[weights>minweight]

        fig = corner.corner(
            samples,
            weights=weights,
            labels=self.psf_result.vparam_names,
            quantiles=(0.16, .5, 0.84),
            bins=20,
            color='k',
            show_titles=True,
            title_fmt='.2f',
            smooth1d=False,
            smooth=True,
            fill_contours=True,
            plot_contours=False,
            plot_density=True,
            use_mathtext=True,
            title_kwargs={"fontsize": 11},
            label_kwargs={'fontsize': 16})
        plt.show()

    def plot_phot(self,method='psf'):
        """
        Plot the photometry

        Parameters
        ----------
        method : str
            psf or aperture
        """
        try:
            if method=='aperture':
                sncosmo.plot_lc(self.aperture_result.phot_cal_table)
            else:
                sncosmo.plot_lc(self.psf_result.phot_cal_table)
        except:
            print('Could not plot phot table for %s method'%method)
            return
        
        plt.show()
        return plt.gcf()