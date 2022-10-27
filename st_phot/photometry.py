import numpy as np
import astropy
import sncosmo
import nestle
from collections import OrderedDict
from copy import copy,deepcopy
import matplotlib.pyplot as plt
import os,sys

from .util import generic_aperture_phot,jwst_apcorr,hst_apcorr,simple_aperture_sum
from .cal import calibrate_JWST_flux,calibrate_HST_flux,calc_jwst_psf_corr

class observation():
    def __init__(self,exposure_fnames,pixel_area_map,sci_ext=1):
        self.exposure_fnames = exposure_fnames if not isinstance(exposure_fnames,str) else [exposure_fnames]
        self.exposures = [astropy.io.fits.open(f) for f in self.exposure_fnames]
        self.data_arr = [im['SCI',sci_ext].data for im in self.exposures]
        self.err_arr = [im['ERR',sci_ext].data for im in self.exposures]
        self.prim_headers = [im[0].header for im in self.exposures]
        self.sci_headers = [im['SCI',sci_ext].header for im in self.exposures]
        self.wcs_list = [astropy.wcs.WCS(hdr) for hdr in self.sci_headers]
        self.n_exposures = len(self.exposures)
        self.telescope = self.prim_headers[0]['TELESCOP']
        self.instrument = self.prim_headers[0]['INSTRUME']

        if 'FILTER' in self.prim_headers[0].keys():
            self.filter = np.unique([hdr['FILTER'] for hdr in self.prim_headers])
        else:
            self.filter = np.unique([hdr['FILTER'] for hdr in self.sci_headers])
        if len(self.filter)>1:
            raise RuntimeError("Each observation should only have one filter.")
        self.filter = self.filter[0]
        if isinstance(pixel_area_map,str):
            self.pams = [im[pixel_area_map].data for im in self.exposures]
        elif isinstance(pixel_area_map,list):
            self.pams = pixel_area_map
        elif isinstance(pixel_area_map,np.ndarray):
            self.pams = [pixel_area_map]*len(self.exposures)
        else:
            raise RuntimeError('Do not recognize your PAM.')

        self.data_arr_pam = [im['SCI',sci_ext].data*pam for im,pam in zip(self.exposures,self.pams)]


    def aperture_photometry(self,sky_location,encircled_energy=70):
        result = {'pos_x':[],'pos_y':[],'aper_bkg':[],'aperture_sum':[],'aperture_sum_err':[],
                  'aper_sum_corrected':[],'aper_sum_bkgsub':[],'annulus_median':[],'exp':[]}
        result_cal = {'flux_cal':[],'flux_cal_err':[],'filter':[],'zp':[],'mag':[],'magerr':[],'zpsys':[],'exp':[]}
        for i in range(self.n_exposures):
            positions = np.atleast_2d(astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[i]))
            if self.telescope=='JWST':
                radius,apcorr,skyan_in,skyan_out = jwst_apcorr(self.exposure_fnames[i],encircled_energy)
                epadu = self.sci_headers[i]['XPOSURE']*self.sci_headers[i]['PHOTMJSR']
            else:
                if self.sci_headers[i]['BUNIT']=='ELECTRON':
                    epadu = 1
                else:
                    epadu = self.prim_headers[i].header['EXPTIME']
                radius,apcorr,skyan_in,skyan_out = hst_apcorr(self.filter,self.instrument,encircled_energy)

            sky = {'sky_in':skyan_in,'sky_out':skyan_out}
            phot = generic_aperture_phot(self.data_arr_pam[i],positions,radius,sky,error=self.err_arr[i],
                                                epadu=epadu)
            for k in phot.keys():
                if k in result.keys():
                    result[k].append(float(phot[k]))
            result['pos_x'].append(positions[0][0])
            result['pos_y'].append(positions[0][1])
            result['aper_sum_corrected'].append(float(phot['aper_sum_bkgsub'] * apcorr))
            result['aperture_sum_err'][-1]*= apcorr
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

        for im in range(self.n_exposures):
            if len(xy_positions)==self.n_exposures:
                xi,yi = xy_positions[im]
            else:
                yi,xi = astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[im])
            centers.append([xi,yi])
            yg, xg = np.mgrid[-1*(fit_width-1)/2:(fit_width+1)/2,
                              -1*(fit_width-1)/2:(fit_width+1)/2].astype(int)
            yf, xf = yg+int(yi+.5), xg+int(xi+.5)
            all_xf.append(xf)
            all_yf.append(yf)

            cutout = self.data_arr_pam[im][xf, yf]
            err = self.err_arr[im][xf, yf]

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


        if len(bounds)!=len(pnames):
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
            pnames = np.append(pnames,['bkg'])
            assert 'bkg' in bounds.keys(),"Must supply bounds for bkg"
            pbounds['bkg'] = bounds['bkg']
        self.nest_psf(pnames,pbounds,cutouts,cutout_errs,all_xf,all_yf,
                        psf_width=fit_width,npoints=npoints,use_MLE=use_MLE,maxiter=maxiter)

    
        result_cal = {'pos_x':[],'pos_y':[],
                      'flux_cal':[],'flux_cal_err':[],'filter':[],
                      'zp':[],'mag':[],'magerr':[],'zpsys':[],'exp':[]}
        model_psf = None
        for i in range(self.n_exposures):
            if fit_flux=='single':
                flux_var = 'flux'
            else:
                flux_var = 'flux%i'%i 

            if fit_centroid=='wcs':
                sky_location = astropy.coordinates.SkyCoord(self.psf_result.best[self.psf_result.vparam_names.index('ra')],
                                                            self.psf_result.best[self.psf_result.vparam_names.index('dec')],
                                                            unit=astropy.units.deg)
                y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,self.wcs_list[i])
            elif fit_centroid=='pixel':
                x = self.psf_result.best[self.psf_result.vparam_names.index('x%i'%i)]
                y = self.psf_result.best[self.psf_result.vparam_names.index('y%i'%i)]
            else:
                x = self.psf_model_list[i].x_0
                y = self.psf_model_list[i].y_0

            xi,yi = centers[i]

            
            flux_sum = simple_aperture_sum(self.psf_model_list[i].data*self.psf_model_list[i].flux,np.atleast_2d([y-yi+self.psf_model_list[i].shape[0]/2,
                                                                                     x-xi+self.psf_model_list[i].shape[1]/2]),10)

            if self.telescope == 'JWST':
                psf_corr,model_psf = calc_jwst_psf_corr(10,self.instrument,self.filter,self.wcs_list[i],psf=model_psf)
                
                flux,fluxerr,mag,magerr,zp = calibrate_JWST_flux(flux_sum*psf_corr,
                    (self.psf_result.errors[flux_var]/self.psf_result.best[self.psf_result.vparam_names.index(flux_var)])*\
                    flux_sum*psf_corr,self.wcs_list[i])
            else:
                raise RuntimeError('not yet implemented for hst')
                # psf_corr = calc_hst_psf_corr(self.instrument,self.filter,self.wcs_list[i])
                flux,fluxerr,mag,magerr,zp = calibrate_HST_flux(self.psf_result.best[self.psf_result.vparam_names.index(flux_var)],
                    self.psf_result.errors[flux_var],self.prim_headers[i],self.sci_headers[i])

            result_cal['pos_x'].append(x)
            result_cal['pos_y'].append(y)
            result_cal['flux_cal'].append(flux)
            result_cal['flux_cal_err'].append(fluxerr)
            result_cal['filter'].append(self.filter)
            result_cal['zp'].append(zp)
            result_cal['mag'].append(mag)
            result_cal['magerr'].append(magerr)
            result_cal['zpsys'].append('ab')
            result_cal['exp'].append(os.path.basename(self.exposure_fnames[i]))

        self.psf_result.phot_cal_table = astropy.table.Table(result_cal)

        print('Finished PSF psf_photometry with median residuals of %.2f'%\
            (100*np.median([self.psf_result.resid_arr[i]/self.psf_result.data_arr[i] for i in range(self.n_exposures)]))+'%')

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

        if 'bkg' in vparam_names:
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
                mflux = self.psf_model_list[i](posx,posy)
                weights = mflux/np.max(mflux)
                if fit_bkg:
                    mflux+=parameters[vparam_names.index('bkg')]
                mflux*=self.pams[i][posx,posy]

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
                #mflux+=res.best[vparam_names.index('bkg')]
                res.data_arr[i]-=res.best[vparam_names.index('bkg')]
            mflux*=self.pams[i][posx,posy]
            resid = res.data_arr[i]-mflux
            all_resid_arr.append(resid)
            
        res.psf_arr = all_mflux_arr
        res.resid_arr = all_resid_arr
        self.psf_result = res
        return 

    def plot_psf_fit(self):
        try:
            temp = self.psf_result.data_arr[0]
        except:
            print('Must fit PSF before plotting.')
            return


        fig,axes = plt.subplots(self.n_exposures,3)
        for i in range(self.n_exposures):
            axes[i][0].imshow(self.psf_result.data_arr[i])
            axes[i][1].imshow(self.psf_result.psf_arr[i])
            axes[i][2].imshow(self.psf_result.resid_arr[i])
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