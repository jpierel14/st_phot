import warnings,os,sys,time,glob,shutil
import numpy as np
import urllib.request
import scipy
import webbpsf

import sncosmo

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy
from astropy import wcs
from astropy.io import fits
from astropy.table import Table,vstack
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from astropy.wcs.utils import skycoord_to_pixel,pixel_to_skycoord
from astropy.nddata import extract_array

import photutils
from photutils import CircularAperture, CircularAnnulus, aperture_photometry
from photutils.psf import EPSFModel
warnings.simplefilter('ignore')
import jwst
from jwst import datamodels
from jwst import source_catalog
from jwst.source_catalog import reference_data
import os

from jwst.datamodels import RampModel, ImageModel
from jwst.pipeline import Detector1Pipeline, Image2Pipeline, Image3Pipeline
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base

from .wfc3_photometry.psf_tools.PSFUtils import make_models
from .wfc3_photometry.psf_tools.PSFPhot import get_standard_psf

__all__ = ['get_jwst_psf','get_hst_psf','get_jwst3_psf','get_hst3_psf','get_jwst_psf_grid',
            'get_jwst_psf_from_grid']

def fancy_background_sub(st_obs,sky_locations=None,pixel_locations=None,width=13,
                                bkg_mode='polynomial',combine_fits=True,do_fit=True,
                                degree=2,h_wht_s = 1,v_wht_s=1,h_wht_p=1,v_wht_p=1,
                                show_plot=False,minval=-np.inf,fudge_center=False,
                                finalmin=-np.inf):
    assert sky_locations is not None or pixel_locations is not None, "Must give skycoord or pixel."
    sys.path.append('/Users/jpierel/CodeBase/manuscript_jupyter/pearls_sn/background_sub')
    import MIRIMBkgInterp
    mbi = MIRIMBkgInterp.MIRIMBkgInterp()
    
    mbi.src_x = (width+2-1)/2
    mbi.src_y = (width+2-1)/2
    mbi.aper_rad = 3 # radius of aperture around source
    mbi.ann_width = 3 # width of annulus to compute interpolation from
    mbi.bkg_mode=bkg_mode # type of interpolation. Options "none","simple","polynomial" 
    mbi.combine_fits = True # use the simple model to attenuate the polynomial model
    mbi.degree = degree # degree of polynomial fit
    mbi.h_wht_s = h_wht_s # horizontal weight of simple model
    mbi.v_wht_s = v_wht_s # vertical weight of simple model
    mbi.h_wht_p = h_wht_p # horizontal weight of polynomial model
    mbi.v_wht_p = v_wht_p # vertical weight of simple model
    if pixel_locations is None and not isinstance(sky_locations,(list,tuple,np.ndarray)):
        sky_locations = [sky_locations]*st_obs.n_exposures
    elif isinstance(pixel_locations[0],(int,float)):
        pixel_locations = [pixel_locations]*st_obs.n_exposures

    final_pixels = []
    nests = []
    for i in range(st_obs.n_exposures):
        if pixel_locations is None:
            if st_obs.n_exposures==1:
                wcs = st_obs.wcs
            else:
                wcs = st_obs.wcs_list[i]
            x,y = wcs.world_to_pixel(sky_locations[i])
            x = int(x)
            y = int(y)
        else:
            x,y = pixel_locations[i]
            x = int(x)
            y = int(y)
        #print(x,y)
        width+=2
        if st_obs.n_exposures==1:
            cutout = st_obs.data[y-int((width-1)/2):y+int((width-1)/2)+1,
                                    x-int((width-1)/2):x+int((width-1)/2)+1]
        else:
            cutout = st_obs.data_arr_pam[i][y-int((width-1)/2):y+int((width-1)/2)+1,
                                    x-int((width-1)/2):x+int((width-1)/2)+1]


        

        cutout[cutout<minval] = np.nan

        if fudge_center:
            init_center = int((width-1)/2)
            #plt.imshow(cutout)
            #plt.show()
            #plt.imshow(cutout[init_center-1:init_center+2,init_center-1:init_center+2])
            #plt.show()
            maxcell = np.argmax(cutout[init_center-1:init_center+2,init_center-1:init_center+2])
            max_ind = np.unravel_index(maxcell,(3,3))
            x,y = np.array([x,y]) + (np.flip(max_ind)-np.array([1,1]))
            #print(x,y)
            if st_obs.n_exposures==1:
                cutout = st_obs.data[y-int((width-1)/2):y+int((width-1)/2)+1,
                                        x-int((width-1)/2):x+int((width-1)/2)+1]
            else:
                cutout = st_obs.data_arr_pam[i][y-int((width-1)/2):y+int((width-1)/2)+1,
                                        x-int((width-1)/2):x+int((width-1)/2)+1]

        final_pixels.append([y,x])
        

            
        if show_plot:
            norm = astropy.visualization.simple_norm(cutout[1:-1,1:-1],invalid=0)
            fig, axes = plt.subplots(1,4,figsize=(12,5))
            axes[0].set_title('image')
            axes[0].imshow(cutout[1:-1,1:-1],origin='lower',norm=norm,cmap='viridis')
        
        # run interpolation
        if not do_fit:
            diff, bkg, mask = mbi.run(cutout)
            
        else:
            (diff, bkg, mask), result_nest = mbi.run_opt(cutout)
            nests.append(result_nest)
        #print(np.nanmedian(diff[0]))
        #print(np.nanmedian(bkg[0]))
        #print(np.nanmedian(mask[0]))
        #print()
        #print()
        width-=2
        for n in range(int((width-1)/2)-1,int((width-1)/2)+2):
            for j in range(int((width-1)/2)-1,int((width-1)/2)+2):
                if diff[0][n][j]<finalmin:
                    diff[0][n][j] = 0
        if st_obs.n_exposures==1:
            st_obs.data[y-int((width-1)/2):y+int((width-1)/2)+1,
                                    x-int((width-1)/2):x+int((width-1)/2)+1] = diff[0]
        else:
            st_obs.data_arr_pam[i][y-int((width-1)/2):y+int((width-1)/2)+1,
                                x-int((width-1)/2):x+int((width-1)/2)+1] = diff[0]

        if show_plot:
            

            

            axes[1].set_title('masked')
            axes[1].imshow(mask[0],origin='lower',norm=norm,cmap='viridis')

            axes[2].set_title('bkg')
            im1 = axes[2].imshow(bkg[0],origin='lower',norm=norm,cmap='viridis')
            divider = make_axes_locatable(axes[2])
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im1, cax=cax, orientation='vertical')
            axes[3].set_title('bkg sub')
            norm = astropy.visualization.simple_norm(diff[0],invalid=0)
            im2 = axes[3].imshow(diff[0],origin='lower',norm=norm,cmap='seismic')
            divider2 = make_axes_locatable(axes[3])
            cax2 = divider2.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im2, cax=cax2, orientation='vertical')
            plt.show()
    st_obs.fancy_background_centers = final_pixels
    return st_obs,nests,mbi

def filter_dict_from_list(filelist,sky_location=None):
    filt_dict = {}
    for f in filelist:
        dat = astropy.io.fits.open(f)
        if sky_location is not None:
            imwcs = astropy.wcs.WCS(dat['SCI',1],dat)
            y,x = skycoord_to_pixel(sky_location,imwcs)
            if not (0<x<dat['SCI',1].data.shape[1] and 0<y<dat['SCI',1].data.shape[0]):
                continue

        filt = dat[0].header['FILTER']
        if filt not in filt_dict.keys():
            filt_dict[filt] = []
        filt_dict[filt].append(f)
    return filt_dict

def get_jwst_psf_grid(st_obs,num_psfs=16):
    inst = webbpsf.instrument(st_obs.instrument)
    inst.filter = st_obs.filter
    inst.detector=st_obs.detector

    grid = inst.psf_grid(num_psfs=num_psfs,all_detectors=False,oversample=4)
    return grid

def get_jwst_psf_from_grid(st_obs,sky_location,grid,psf_width=101):

    grid.oversampling = 1
    psf_list = []
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        grid.x_0 = x
        grid.y_0 = y
       
        xf, yf = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))

        psf = np.array(grid(xf,yf)).astype(float)
        epsf_model = photutils.psf.FittableImageModel(psf,normalize=True,oversampling=4)
        psf_list.append(epsf_model)
    return psf_list

def get_jwst_psf(st_obs,sky_location,num_psfs=16,psf_width=101):
    inst = webbpsf.instrument(st_obs.instrument)
    inst.filter = st_obs.filter
    inst.detector=st_obs.detector
    inst.pixelscale = st_obs.pixel_scale

    grid = inst.psf_grid(num_psfs=num_psfs,all_detectors=False,oversample=4)
    psf_list = []
    grid.oversampling=1
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)

        grid.x_0 = x
        grid.y_0 = y
        xf, yf = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))
        psf = np.array(grid(xf,yf)).astype(float)
        
        epsf_model = photutils.psf.FittableImageModel(psf,normalize=True,oversampling=4)
        psf_list.append(epsf_model)
    return psf_list

def get_jwst3_psf(st_obs,sky_location,num_psfs=16,psf_width=101):
    #psfs = get_jwst_psf(st_obs,sky_location,num_psfs=num_psfs,psf_width=psf_width)
    grid = get_jwst_psf_grid(st_obs,num_psfs=num_psfs)
    grid.oversampling = 1 
    psfs = []
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        x,y = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        grid.x_0 = x
        grid.y_0 = y
       
        xf, yf = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))

        psf = np.array(grid(xf,yf)).astype(float)
        epsf_model = photutils.psf.FittableImageModel(psf,normalize=True,oversampling=1)
        psfs.append(epsf_model)

    outdir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'temp_%i'%np.random.randint(0,1000))
    os.mkdir(outdir)
    #print(outdir)
    try:
        out_fnames = []
        for i,f in enumerate(st_obs.exposure_fnames):
            #print(f)
            dat = fits.open(f)
            dat['SCI',1].data[np.isnan(dat['SCI',1].data)] = 0
            #xf, yf = np.mgrid[0:dat['SCI',1].data.shape[0],0:dat['SCI',1].data.shape[1]].astype(int)
            norm = astropy.visualization.simple_norm(dat['SCI',1].data,invalid=0,min_cut=-.15,max_cut=.3)
            #print(np.max(dat['SCI',1].data))
            #plt.imshow(dat[1].data,norm=norm)
            #plt.show()
            imwcs = wcs.WCS(dat['SCI',1])
            y,x = skycoord_to_pixel(sky_location,imwcs)
            
            #print(x,y,pixel_to_skycoord(x,y,imwcs))
            if False:
                newx = dat[1].header['NAXIS1']*4
                newy = dat[1].header['NAXIS2']*4
                dat[1].header['NAXIS1'] = newx
                dat[1].header['NAXIS2'] = newy
                old_wcs = wcs.WCS(dat[1])
                #print(old_wcs)
                new_wcs = old_wcs[::.25,::.25].to_header()
                for k in ['PC1_1', 'PC1_2','PC2_1','PC2_2']:
                    new_wcs[k]/=4


                for key in new_wcs.keys():
                    if len(key)>0:
                        #dm_fits[i].header[key+'A'] = dm_fits[i].header[key]
                        #if not (self.do_driz or ('CRPIX' in key or 'CTYPE' in key)):
                        if 'CTYPE' not in key:
                            if key.startswith('PC') and key not in dat[1].header.keys():
                                dat[1].header.set(key.replace('PC','CD'),value=new_wcs[key])
                            elif key in dat[1].header:
                                dat[1].header.set(key,value=new_wcs[key])
                                #else:
                                #    dm_fits[i].header.set(key,value='TWEAK')
                            #else:
                            #   print(key)
                            #   sys.exit()
                dat[1].header['PIXAR_A2'] = dat[1].header['PIXAR_A2']/16
            #print(newx,newy)
            #dat['SCI',1].data = np.zeros((newx,newy))
            imwcs = wcs.WCS(dat['SCI',1])
            #print(imwcs)
            y,x = skycoord_to_pixel(sky_location,imwcs)
            #print(x,y,pixel_to_skycoord(x,y,imwcs),dat['SCI',1].data.shape)

            xf, yf = np.mgrid[0:dat['SCI',1].data.shape[0],0:dat['SCI',1].data.shape[1]].astype(int)
            psfs[i].x_0 = x
            psfs[i].y_0 = y
            
        
            dat['SCI',1].data = psfs[i](xf,yf)
            dat['SCI',1].data/=st_obs.pams[i]#scipy.ndimage.zoom(st_obs.pams[i],4)
            #print(np.max(dat['SCI',1].data))
            #dat['SCI',1].data[dat['SCI',1].data>.005] = 10000
            #plt.imshow(dat[1].data[xf,yf],vmin=0,vmax=.005)

            #plt.show()
            #sys.exit()
            
            dat['ERR',1].data = np.ones((1,1))
            dat['VAR_RNOISE',1].data = np.ones((1,1))
            dat['VAR_POISSON',1].data = np.ones((1,1))
            dat['VAR_FLAT',1].data = np.ones((1,1))
            dat['DQ',1].data = np.zeros(dat[1].data.shape)
            dat.writeto(os.path.join(outdir,os.path.basename(f)),overwrite=True)
            out_fnames.append(os.path.join(outdir,os.path.basename(f)))
        #sys.exit()
        asn = asn_from_list.asn_from_list(out_fnames, rule=DMS_Level3_Base, 
            product_name='temp_psf_cals')
        
        with open(os.path.join(outdir,'cal_data_asn.json'),"w") as outfile:
            name, serialized = asn.dump(format='json')
            outfile.write(serialized)
        pipe3 = Image3Pipeline()
        pipe3.output_dir = outdir
        pipe3.save_results = True
        pipe3.tweakreg.skip = True
        pipe3.outlier_detection.skip = True
        pipe3.skymatch.skip = True
        pipe3.source_catalog.skip = True
        #pipe3.resample.output_shape = (newx,newy)
        pipe3.outlier_detection.save_results = False
        #pipe3.resample.pixel_scale = np.sqrt(dat[1].header['PIXAR_A2'])
        #pipe3.resample.pixel_scale_ratio = .25
        pipe3.run(os.path.join(outdir,'cal_data_asn.json'))
        dat = fits.open(os.path.join(outdir,'temp_psf_cals_i2d.fits'))
        imwcs = wcs.WCS(dat['SCI',1])
        level3 = dat[1].data
        level3[np.isnan(level3)] = 0 
        #print(np.max(level3))
        #sys.exit()
        y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        mx,my = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))
        
        level3_psf = photutils.psf.FittableImageModel(level3[mx,my],normalize=True, 
                                                      oversampling=4)
        shutil.rmtree(outdir)
    except RuntimeError:
        print('Failed to create PSF model')
        shutil.rmtree(outdir)
    return level3_psf

def get_hst_psf(st_obs,sky_location,psf_width=25):
    grid = make_models(get_standard_psf(os.path.join(os.path.abspath(os.path.dirname(__file__)),
            'wfc3_photometry/psfs'),st_obs.filter,st_obs.detector))[0]
    psf_list = []
    for i in range(st_obs.n_exposures):
        imwcs = st_obs.wcs_list[i]
        y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        psfinterp = grid._calc_interpolator(int(x), int(y))
        _psf_interp = psfinterp(grid._xidx, grid._yidx)
        psfmodel = photutils.psf.FittableImageModel(_psf_interp,
                                      oversampling=grid.oversampling)
        psfmodel.x_0 = x#int(x)
        psfmodel.y_0 = y#int(y)
        psf_list.append(psfmodel)

        #yg, xg = np.mgrid[-1*(psf_width-1)/2:(psf_width+1)/2,-1*(psf_width-1)/2:(psf_width+1)/2].astype(int)
        #yf, xf = yg+int(y+.5), xg+int(x+.5)
        #yf, xf = yg+int(np.round(y)), xg+int(np.round(x))
        #psf = np.array(psfmodel(xf,yf)).astype(float)
        #plt.imshow(psf)
        #plt.show()
        #continue
        #print(x,y)
        
        #epsf_model = EPSFModel(psf)
        #psf_list.append(epsf_model)
    return psf_list

def get_hst3_psf(st_obs,sky_location,psf_width=25):
    from drizzlepac import astrodrizzle
    psfs = get_hst_psf(st_obs,sky_location,psf_width=psf_width)

    outdir = os.path.join(os.path.abspath(os.path.dirname(__file__)),'temp_%i'%np.random.randint(0,1000))
    os.mkdir(outdir)
    try:
        out_fnames = []
        for i,f in enumerate(st_obs.exposure_fnames):
            dat = fits.open(f)
            newx = dat[1].header['NAXIS1']*4
            newy = dat[1].header['NAXIS2']*4
            
            old_wcs = wcs.WCS(dat[1],dat)
            new_wcs = old_wcs[::.25,::.25].to_header()
            for k in ['PC1_1', 'PC1_2','PC2_1','PC2_2']:
                new_wcs[k]/=4


            for key in new_wcs.keys():
                   if len(key)>0:
                       #dm_fits[i].header[key+'A'] = dm_fits[i].header[key]
                       #if not (self.do_driz or ('CRPIX' in key or 'CTYPE' in key)):
                        if 'CTYPE' not in key:
                            if key.startswith('PC') and key not in dat[1].header.keys():
                                dat[1].header.set(key.replace('PC','CD'),value=new_wcs[key])
                            elif key in dat[1].header:
                                dat[1].header.set(key,value=new_wcs[key])
                            #else:
                            #    dm_fits[i].header.set(key,value='TWEAK')
                            
            dat[1].header['IDCSCALE'] = dat[1].header['IDCSCALE']/4
            dat['SCI',1].data = np.zeros((newy,newx))
            y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,wcs.WCS(dat[1],dat))
            psf2 = photutils.psf.FittableImageModel(psfs[i].data,normalize=True, 
                                                      oversampling=1)
            psf2.x_0 = x
            psf2.y_0 = y
            x = int(x+.5)
            y = int(y+.5)
            
            gx, gy = np.mgrid[0:newx,0:newy].astype(int)
            dat[1].data = psf2.evaluate(gx,gy,psf2.flux.value,psf2.x_0.value,psf2.y_0.value,
                                        use_oversampling=False)
            dat[1].data[dat[1].data<0] = 0

            dat[1].data/=scipy.ndimage.zoom(st_obs.pams[0].T,4)

            dat['DQ',1].data = np.zeros((newx,newy)).astype(int)
            dat['ERR',1].data = np.ones((newx,newy))
            print(dat)
            dat = dat[:4]
            print(dat)
            dat.writeto(os.path.join(outdir,os.path.basename(f)),overwrite=True)
            out_fnames.append(os.path.join(outdir,os.path.basename(f)))
            
        

        astrodrizzle.AstroDrizzle(','.join(out_fnames),output=os.path.join(outdir,'temp_psf'),
                            build=True,median=False,skysub=False,
                            driz_cr_corr=False,final_wht_type='ERR',driz_separate=False,
                            driz_cr=False,blot=False,clean=True,
                            final_outnx=int(1014*4),final_outny=int(1014*4))
        
        try:
            dat = fits.open(glob.glob(os.path.join(outdir,'temp_psf_drz.fits'))[0])
        except:
            dat = fits.open(glob.glob(os.path.join(outdir,'temp_psf_drc.fits'))[0])
        imwcs = wcs.WCS(dat[1],dat)
        y,x = skycoord_to_pixel(sky_location,imwcs)
        level3 = dat[1].data
        level3[np.isnan(level3)] = 0 
        print(level3.shape)
        y,x = astropy.wcs.utils.skycoord_to_pixel(sky_location,imwcs)
        my,mx = np.meshgrid(np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(x+.5),
                            np.arange(-4*psf_width/2,psf_width/2*4+1,1).astype(int)+int(y+.5))
        print(x,y,np.max(mx),np.max(my))
        level3_psf = photutils.psf.FittableImageModel(level3[mx,my],normalize=True, 
                                                      oversampling=4)
        
        shutil.rmtree(outdir)
    except RuntimeError:
        print('Failed to create PSF model')
        shutil.rmtree(outdir)
    return level3_psf


def jwst_apcorr(fname,ee=70,alternate_ref=None):
    sc = source_catalog.source_catalog_step.SourceCatalogStep()
    if alternate_ref is not None:
        fname = alternate_ref
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
                                    'uvis-encircled-energy/_documents/wfc3uvis2_aper_007_syn.csv','wfc3uvis2_aper_007_syn.csv')
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
