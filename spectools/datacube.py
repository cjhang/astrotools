"""This is the core class that handle datacubes from IFU observation, including
function to do spectrum fitting of the whole cube and generate the results into
files

:TODO: add a visual tools to handle all the manipulation

"""

import os
import re
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits

import spectools
from spectools.utils import ppxf, miles_util, ppxf_util, capfit
from spectools.spectrum import Spectrum

class Datacube(object):
    """ The basic class to handle datacube
    """

    def __init__(self, wave=None, z=None, flux=None, noise=None,
                 redcorr=None, instru_sigma=None):
        self.wave = wave
        self.z = z
        self.flux = flux
        self.noise = noise
        self.redcorr = redcorr
        self.instru_sigma = instru_sigma

    def __getitem__(self, xy):
        '''shortcut for spaxel requiring
        '''
        return self.getspaxel(x=xy[0], y=xy[1])

    def getspaxel(self, x=None, y=None, ra=None, dec=None):
        """get different spaxel either from index or from ra and dec

        Parameters
        ----------
        x : int
            index from the x coordiate
        y : int 
            index from the y coordiate
        ra : float
            ra of the spaxel
        dec : float
            dec of the spaxel

        Returns
        -------
        Spectrum : class
            `spectools.spectrtrum.Spectrum`
        """
        spaxel = Spectrum()
        spaxel.wave = self.wave
        if ra is not None and dec is not None:
            sp_coor = coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
            radec_xy = wcs.utils.skycoord_to_pixel(sp_coor, self.wcs_cude)
            x = np.round(radec_xy[1]).astype(int)
            y = np.round(radec_xy[0]).astype(int)
        spaxel.flux = self.flux[:, x, y]
        spaxel.redcorr = self.redcorr
        spaxel.noise = self.noise[:, x, y]
        spaxel.z = self.z
        spaxel.instru_sigma = self.instru_sigma
        return spaxel

    def fitmap(self, filename, mask=None, directory='/', auto=False, 
               save_map=True, save_cube=False, quiet=False, **kwargs):
        """generate all the fitted map and datacube for the raw spectrum 
        datacube

        Parameters
        ----------
        filename : str
            the filename of the output file.
        mask : array_like, optional
            the spaxel for fitting.
        directory : str, optional
            the directory of output files, default the current working 
            directory.
        auto : bool, optional
            choosing the best fit models automatically.
        save_map : bool, optional
            saving the flux map of all the fitted emission lines, default is 
            True.
        save_cube : bool, optional
            saving the fitted spectrum with different components into file, 
            default is True.
        quiet : bool, optional
            set it to True to suppress the output, default False.
        kwargs : optional
            the additional parameters that pass to ppxf

        Returns
        -------
        The status of the fitting processes

        TODO:
            make the tempalte as input parameter

        """
        npix, naxis1, naxis2 = self.flux.shape
        if mask is None:
            mask = np.full((naxis1, naxis2), False)
        
        lam_range_gal = np.array([np.min(self.wave), np.max(self.wave)])/(
                                1+self.z)
        velscale = 299792.458 * np.log(self.wave[1]/self.wave[0])  # km/s
        FWHM_gal = 2.7

        # read the template
        package_path = os.path.dirname(os.path.realpath(spectools.__file__))
        miles = miles_util.miles(package_path + '/data/miles_models/Mun1.30*.fits', 
                                 velscale, FWHM_gal)
        
        gas_templates, gas_names, line_wave = ppxf_util.emission_lines(
                    miles.log_lam_temp, lam_range_gal, FWHM_gal, quiet=quiet,
                    **kwargs)

        # define the class to store fitted data
        fitmap = FitMap(gas_names, (naxis1, naxis2))
        fitcube = FitCube(self.flux.shape)
        fitcube.wave = self.wave
        fitcube.flux = self.flux

        for x in range(naxis1):
            for y in range(naxis2):
                if mask[x, y]:
                    continue
                if quiet==2 or quiet==0:
                    print('coordinate:', [x, y])
                sp = self[x, y]
                if auto:
                    try:
                        try: 
                            if sp.stellarcontinuum is None:
                                pp = sp.ppxf_fit(mode='kinematics', quiet=quiet,
                                                 **kwargs)
                            # fitting with strong AGN
                            pp1 = sp.ppxf_fit(mode='emline', quiet=quiet, 
                                              broad_balmer=800)
                            pp2 = sp.ppxf_fit(mode='emline', quiet=quiet, 
                                              broad_balmer=800, broad_O3=600)
                        except:
                            # fitting with weak AGN
                            if not quiet:
                                print('Change emline with fewer emission lines!')
                            pp1 = sp.ppxf_fit(mode='emline', quiet=quiet, 
                                              fewer_lines=True)
                            pp2 = sp.ppxf_fit(mode='emline', quiet=quiet, 
                                              fewer_lines=True, broad_O3=600)
                    except KeyboardInterrupt:
                        sys.exit()
                    except: # for failed fitting
                        if not quiet:
                            print("Fitting failed!")
                        continue

                    F = (pp1.chi2_orig - pp2.chi2_orig)*(pp2.vars_num - pp2.params_num ) \
                            / (pp2.params_num - pp1.params_num) / pp2.chi2_orig
                    p_value = 1 - stats.f.cdf(F, pp2.params_num - pp1.params_num, 
                                                 pp2.vars_num - pp2.params_num)
                    if (p_value < 0.05) and ((pp2.chi2 - 1) < (pp1.chi2 - 1)) \
                            and np.any(pp2.gas_flux[-2:]/pp2.gas_flux_error[-2:] > 3):
                        pp = pp2
                        if not quiet:
                            print('Prefer broad [O III]')
                    else:
                        pp = pp1
                    if not quiet:
                        print('p_value:', p_value, 'fit1 chi2:', pp1.chi2, 
                              'fit2 chi2:', pp2.chi2)
                else:
                    if sp.stellarcontinuum is None:
                        pp = sp.ppxf_fit(mode='kinematics', quiet=quiet, 
                                         **kwargs)
                    pp = sp.ppxf_fit(mode='emline', quiet=quiet, **kwargs)
                dwave = np.roll(pp.lam, -1) - pp.lam
                dwave[-1] = dwave[-2] # fix the bad point introduced by roll
                flux = dwave @ pp.matrix * pp.weights * pp.flux_scale
                flux_err = dwave @ pp.matrix \
                           * capfit.cov_err(pp.matrix / pp.noise[:, None])[1] \
                           * pp.flux_scale
                
                gas_flux = dict(zip(pp.gas_names, flux))
                gas_flux_err = dict(zip(pp.gas_names, flux_err))
                v, sigma = np.transpose(np.array(pp.sol)[pp.component.tolist()])
                rel_v = dict(zip(pp.gas_names, v - 299792.485 * np.log(1+pp.z)))
                sigma = dict(zip(pp.gas_names, sigma))
                for name in pp.gas_names:
                    fitmap[name][:, x, y] = gas_flux[name], \
                                            gas_flux_err[name],\
                                            rel_v[name], sigma[name]
                if save_cube:
                    wave_mask = ((self.wave >= pp.lam[0]) 
                                 & (self.wave <= pp.lam[-1]))
                    cube_fits = pp.matrix @ (pp.weights * pp.flux_scale)
                    fitcube[wave_mask, x, y] = cube_fits
                    # # balmer, forbidden, and broad lines
                    # for comp_n in [0, 1, 2]: 
                        # wave_mask = ((self.wave >= pp.lam[0]) 
                                     # & (self.wave <= pp.lam[-1]))
                        # comp_select = np.where(pp.component == comp_n)
                        # cube_comp = pp.matrix[:, comp_select] @ (
                                    # pp.weights[comp_select] * pp.flux_scale)
                        # datacube[wave_mask, x, y, comp_n] = cube_comp[:, 0]
        if save_map:
            fitmap.save(filename, directory=directory)
        if save_cube:
            fitcube.save(filename, directory=directory)

class FitMap(object):

    """The class to store fitted results of Datacube"""

    def __init__(self, gas_names, shapes):
        """store flux and kinemtic data for every emission line

        Parameters
        ----------
        gas_names : array_like
            the names of every emission line
        shapes : array_like
            the dimension of the map
        comments : str, optional
            comments need to be included in the fits header

        """
        self.gas_names = gas_names
        
        # each emission lines should include flux, flux_err, v, sigma
        self.data = dict(zip(gas_names, 
                        np.zeros((len(gas_names), 4, *shapes))))
        
    def __getitem__(self, gas_name):
        """access the data like numpy array

        Parameters
        ----------
        gas_names : str
            the key of the multimap dictionary
        """
        return self.data[gas_name]

    def save(self, filename, auther_name=None, comments=None, directory=None):
        """save the fitted map into fits file

        Parameters
        ----------
        filename : str
            the name of the saved file
        auther_name : str, optional
            the name of the auther
        comments : str, optional
            the additional comments imformation added to the fits header
        directory : str, optional
            the desired destination to store the fitted data

        """
        hdr = fits.Header()
        hdr['AUTHER'] = auther_name
        if comments is None:
            comments = "Fitting emission lines with broad lines"
        if directory is None:
            directory = './'
        hdr['COMMENT'] = comments
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdu_list = [primary_hdu]
        for name in self.gas_names:
            hdu_list.append(fits.ImageHDU(self.data[name], name=name))
        hdus = fits.HDUList(hdu_list)
        hdus.writeto('{}/{}-maps.fits'.format(directory, filename), 
                     overwrite=True)

class FitCube(object):

    """the class to store the fitted spectrum data"""

    def __init__(self, shapes):
        """store the two dimensional fitted spectrum

        Parameters
        ----------
        shapes : array_like
            the dimensions of the integrated field units


        """
        self.wave = None
        self.flux = np.zeros(shapes)
        self.data = np.zeros(shapes)

    def __getitem__(self, index):
        """access the fitted spectrum like ndarray

        Parameters
        ----------
        index : array_like
            slice index of the data of FitCube

        """
        return self.data[index]

    def __setitem__(self, index, value):
        """set values
        
        Parameters
        ----------
        index : array_like
            slice index of the self.data
        value : array_like
            new value assigned to self.data
        """
        self.data[index] = value
        

    def save(self, filename, auther_name=None, comments=None, directory=None):
        """save the fitted spectrum

        Parameters
        ----------
        auther_name : str, optional
            the name of the auther
        comments : str, optional
            the additional comments imformation added to the fits header
        directory : str, optional
            the desired destination to store the fitted data

        """
        hdr = fits.Header()
        hdr['AUTHER'] = auther_name
        if comments is None:
            comments = "Fitting emission lines with broad lines"
        if directory is None:
            directory = './'
        hdr['COMMENT'] = comments
        primary_hdu = fits.PrimaryHDU(header=hdr)
        hdu_cubes_wave = fits.ImageHDU(self.wave, name='wave')
        hdu_cubes_flux = fits.ImageHDU(self.flux, name='flux')
        #hdu_cubes_emlines = fits.ImageHDU(self.emlines. name='emlines')
        hdu_cubes_fits = fits.ImageHDU(self.data, name='fits')
        hdu_cubes = [primary_hdu, hdu_cubes_wave, hdu_cubes_flux, 
                     hdu_cubes_fits]
        hdu_cubes = fits.HDUList(hdu_cubes)
        hdu_cubes.writeto('{}/{}-cubes.fits'.format(directory, filename), 
                          overwrite=True)

