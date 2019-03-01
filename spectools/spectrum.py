"""
The basic class to handle spectrum, including full spectrum fitting based on
pPXF developed by Cappellari Michele Cappellari [1]_. 

.. [1] Cappellari, M. 2017, MNRAS, 466, 798
"""

import os
import re
import numpy as np
from matplotlib import pyplot as plt
import spectools
from  .utils import ppxf, miles_util, ppxf_util

class Spectrum(object):
    """The basic class for spectrum
    """

    def __init__(self, wave=None, z=None, flux=None, noise=None, 
                 instru_sigma=None, redcorr=None):
        """
        Parameters
        ----------
        wave : array_like
            wave of the spectrum
        z : float
            bestfit redshift of the spectrum
        flux : array_like
            flux of the spectrum, same shape as wave
        noise : array_like
            noise of the spectrum, same shape as wave
        instru_sigma : float
            instrument sigma
        redcorr : array_like
            dust redding correction of the flux, same shape like flux
            it usefule for Galaxy forground extinction correction

        Attributes
        ----------
        model : array_like
            best fitting model of the spectrum
        stellarcontinuum : array_like
            best fitting stellar continuum with emission lines masked
        emlines: array_like
            Gaussian or multiple Gaussian modeled emmission lines
        emline_base: array_like
            Base of the emission lines, same like the continuum
        residual : array_like
            residual of the spectrum after subtracting the model
        
        Methods
        -------
        read(filename)
            read spectrum directly from file
        plot()
            visualize the spectrum
        ppxf_fit(mode='kinematics')
            Full spectrum fitting based on the pPXF code
        """
        self.wave = wave
        self.z = z
        self.flux = flux
        self.noise = noise
        self.fitted = False
        #self.dx = None
        self.instru_sigma = instru_sigma
        self.redcorr = redcorr
        
        # model data
        self.model = None
        self.stellarcontinuum = None
        self.emlines = None
        self.emline_base = None
        self.residual = None
        #self.sigma = self.psf/2.355

    def read(self, filename):
        '''read from fits file with standard header
        
        Params:
            filename:
        '''
        pass

    def plot(self, waveRange=None, restWave=False, showFlux=True, ax=None,
             showModel=False, showEmlines=False, showContinuum=False,
             showResidual=False):
        """Visualize the spectrum with original data and fitted data

        Parameters
        ----------
        waveRange : list or None
            The wavelength window of the visualization, None for full wavelength
        restWave : bool
            Option for whether change to wavelength to restframe
        showFlux : bool
            display the original flux of the spectrum
        ax : class or None
            The subplot of matplotlib figure
        showModel : bool
            display the best fitting model
        showEmlines : bool
            show fitted emlines lines
        showContinuum : bool
            show best fitting stellar continuum
        showResidual : bool
            residual = flux - model
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        if restWave and self.z:
            wave = self.wave / (1 + self.z)
            ax.set_xlabel("wavelength (rest frame)")
        else:
            wave = self.wave
            ax.set_xlabel("wavelength")
        if showFlux:
            ax.step(wave, self.flux, label='flux', color='k', lw=0.5)
        if self.fitted:
            if showModel:
                ax.plot(self.wave_fit, self.model, label='model', color='r', lw=1)
            if showEmlines:
                ax.plot(self.wave_fit, self.emlines, label='emlines', color='b', lw=1)
            if showContinuum:
                ax.plot(self.wave_fit, self.stellarcontinuum, label='stellar continuum',
                        color='g', lw=1)
            if showResidual:
                ax.step(self.wavefit, self.residual-0.5, label='residual', 
                        where='mid', color='0.5', lw=0.5)
        if waveRange:
            wave_window = (wave > waveRange[0]) & (wave < waveRange[1]) 
            ax.set_xlim(waveRange)
            ax.set_ylim(self.flux[wave_window].min(), 
                        self.flux[wave_window].max())
        ax.set_ylabel('Flux')
        ax.legend()

        if not ax:
            return fig

    def ppxf_fit(self, tie_balmer=False, limit_doublets=False, quiet=False, 
                 mode='population', broad_balmer=None, broad_O3=None, 
                 fewer_lines=False):
        """fitting the spectrum using ppxf
        
        Parameters
        ----------
        tie_balmer : bool
            Fix the flux ratio and kinematics of Balmer lines
        quiet : bool
            Set to Ture to suppress all the output messages
        mode: str
            Three mode can be applied to the fitting:

            kinematics  
                mainly used to get the kinematics of the stellar continuum

            population
                fitting the emission lines and continuum at the same time

            emline
                a more accurate way to fit the emission lines, it will firstly 
                subtract the stellar continuum and then fitting every emission
                individually
        broad_balmer : float or None
            Add broad component to the Balmer lines, with the value used as the
            minimal velocity of the borad Gaussian line profile
        broad_O3 : float or None
            The same as broad_balmer, but adding broad component to the 
            [O III]5007 line
        fewer_lines : bool
            Set to true to use few lines when calling pPXF, useful for 
            elliptical red galaxies with weak emission lines

        Returns
        -------
        class
            `ppxf.ppxf`

        **TODO**
            * add the miles library as the option of the function
            *  

        """
        # Only use the wavelength range in common between galaxy and stellar library.
        wave_mask = (self.wave > 3540*(1+self.z)) & (self.wave < 7400*(1+self.z))
        self.flux_scale = np.ma.median(self.flux[wave_mask])
        flux = self.flux[wave_mask]/self.flux_scale
        if self.redcorr is not None:
            flux = flux * self.redcorr[wave_mask]
        wave = self.wave[wave_mask]
        noise = self.noise[wave_mask]/self.flux_scale
        if not np.all((noise > 0) & np.isfinite(noise)) and not quiet:
            print('noise:', noise)
            print('flux_scale:', self.flux_scale)

        # noise = np.full_like(flux, 0.0166)
        c = 299792.485
        # define the dispersion of one pixel
        velscale = c*np.log(wave[1]/wave[0]) 
        FWHM_gal = self.instru_sigma
        
        # load the templates
        package_path = os.path.dirname(os.path.realpath(spectools.__file__))
        miles = miles_util.miles(package_path + '/data/miles_models/Mun1.30*.fits', 
                                 velscale, FWHM_gal)
        reg_dim = miles.templates.shape[1:]
        stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
        
        # before fitting
        dv = c*(miles.log_lam_temp[0] - np.log(wave[0]))
        vel = c*np.log(1+self.z)
        start = [vel, 180.]
        
        if mode == 'population':
            regul_err = 0.013 # Desired regularization error
            lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1+self.z)
            
            gas_templates, gas_names, line_wave = ppxf_util.emission_lines(
                    miles.log_lam_temp, lam_range_gal, FWHM_gal,
                    tie_balmer=tie_balmer, limit_doublets=limit_doublets, 
                    broad_balmer=broad_balmer, broad_O3=broad_O3,
                    fewer_lines=fewer_lines)
            
            templates = np.column_stack([stars_templates, gas_templates])
            n_temps = stars_templates.shape[1]
            
            # Balmer lines start with contain letter
            n_balmer = np.sum([re.match('^[a-zA-Z]+', a) is not None for a in gas_names]) 
            
            # forbidden lines contain "["
            # n_forbidden = np.sum(["[" in a for a in gas_names])
            n_forbidden = np.sum([re.match('^\[[a-zA-Z]+', a) is not None 
                                    for a in gas_names])
            
            component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden 
            component_n = 3
            
            # for stars + balmer + forbidden lines
            moments = [4, 2, 2]
            start = [start, start, start]
            inf = 1e8
            bounds = [[[-inf, inf], [0, 500], [-inf, inf], [-inf, inf]], 
                      [[-inf, inf], [0, 500]], 
                      [[-inf, inf], [0, 500]]]
            
            if broad_balmer is not None:
                moments.append(2)
                start.append([vel, broad_balmer])
                # balmer up to 10000
                bounds.append([[-inf, inf], [broad_balmer, 10000]])
                # broad lines contain "{*}"
                n_balmer_broad = np.sum([re.match('^_[a-zA-Z]+', a) is not None 
                                            for a in gas_names]) 
                component = component + [component_n]*n_balmer_broad
                component_n = component_n + 1
            
            if broad_O3 is not None:
                moments.append(2)
                start.append([vel, broad_O3])
                bounds.append([[-inf, inf], [broad_O3, 2000]])
                n_forbidden_broad = np.sum([re.match('^_\[[a-zA-Z]+', a) is not None 
                                              for a in gas_names])
                component = component + [component_n]*n_forbidden_broad
            
            gas_component = np.array(component) > 0
            gas_reddening = 0 if tie_balmer else None
            # start fitting
            mask = None
            pp = ppxf.ppxf(templates, flux, noise, velscale, start,
                      plot=False, moments=moments, degree=12, mdegree=0, 
                      vsyst=dv, lam=wave, clean=False, regul=1/regul_err, 
                      reg_dim=reg_dim, mask=mask, component=component, 
                      quiet=quiet, gas_component=gas_component, 
                      gas_names=gas_names, gas_reddening=gas_reddening)
       
        elif mode == "kinematics":
            templates = stars_templates
            lamRange_temp = (np.exp(miles.log_lam_temp[0]), 
                            np.exp(miles.log_lam_temp[-1]))
            flux = np.ma.array(flux)
            bad_mask = np.where((flux.mask == True) 
                                & (flux > 10*np.ma.median(flux)))[0]
            goodpixels = ppxf_util.determine_goodpixels(
                            np.log(wave), lamRange_temp, self.z, 
                            broad_balmer=broad_balmer,
                            broad_O3=broad_O3)
            # remove the masked data from the goodpixels
            for i in bad_mask:
                try:
                    goodpixels.remove(i)
                except:
                    continue

            flux = flux.filled(0)
            pp = ppxf.ppxf(templates, flux, noise, velscale, start,
                    goodpixels=goodpixels, plot=False, moments=4, 
                    degree=12, vsyst=dv, clean=False, lam=wave, quiet=quiet)
            
            # store the fitting results
            self.fitted = True
            self.stellarcontinuum = np.full_like(self.flux, 0)
            self.stellarcontinuum[wave_mask] = pp.bestfit * self.flux_scale
            self.wave_fit = pp.lam
            self.window_fit = ((self.wave <= self.wave_fit.max()) 
                                & (self.wave >= self.wave_fit.min()))
        
        elif mode == 'emline':
            flux = (self.flux[wave_mask] - self.stellarcontinuum[wave_mask])
            if self.flux_scale:
                flux = flux / self.flux_scale
            if self.redcorr:
                flux = flux * self.redcorr
            lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1+self.z)
            gas_templates, gas_names, line_wave = ppxf_util.emission_lines(
                    miles.log_lam_temp, lam_range_gal, FWHM_gal, fewer_lines=fewer_lines, 
                    tie_balmer=tie_balmer, limit_doublets=limit_doublets, 
                    broad_balmer=broad_balmer, broad_O3=broad_O3, quiet=quiet)
            
            # Balmer lines start with contain letter
            n_balmer = np.sum([re.match('^[a-zA-Z]+', a) is not None for a in gas_names]) 
            
            # forbidden lines contain "["
            # n_forbidden = np.sum(["[" in a for a in gas_names])
            n_forbidden = np.sum([re.match('^\[[a-zA-Z]+', a) is not None 
                                    for a in gas_names])
            component = [0]*n_balmer + [1]*n_forbidden 
            component_n = 2
            
            # for stars + balmer + forbidden lines
            moments = [2, 2]
            start = [start, start]
            inf = 1e6
            bounds = [[[-inf, inf], [0, 600]], 
                      [[-inf, inf], [0, 600]]]
            if broad_balmer is not None:
                moments.append(2)
                start.append([vel, broad_balmer])
                # balmer up to 10000
                bounds.append([[-inf, inf], [broad_balmer, 10000]])
                # broad lines contain "_["
                n_balmer_broad = np.sum([re.match('^_[a-zA-Z]+', a) is not None 
                                            for a in gas_names]) 
                component = component + [component_n]*n_balmer_broad
                component_n = component_n + 1
            
            if broad_O3 is not None:
                moments.append(2)
                start.append([vel, broad_O3])
                bounds.append([[-inf, inf], [broad_O3, 2000]])
                n_forbidden_broad = np.sum([re.match('^_\[[a-zA-Z]+', a) is not None 
                                              for a in gas_names])
                component = component + [component_n]*n_forbidden_broad
            gas_component = np.array(component) >= 0
            gas_reddening = 0 if tie_balmer else None
            
            # start fitting
            mask = None
            pp = ppxf.ppxf(gas_templates, flux, noise, velscale, start, bounds=bounds,
                      plot=False, moments=moments, degree=-1, mdegree=0, 
                      vsyst=dv, lam=wave, clean=False, mask=mask, component=component, 
                      quiet=quiet, gas_component=gas_component, 
                      gas_names=gas_names, gas_reddening=gas_reddening)
        
        # wrap relevant value into returned class 
        pp.flux_scale = self.flux_scale
        pp.z = self.z
        pp.mode = mode
        # pp.var_num = len(pp.lam)
        # pp.para_num = len(np.concatenate(start))
        pp.chi2_orig = pp.chi2 * (pp.vars_num - pp.params_num)
        
        # seperating different components from the model
        if False:#len(pp.gas_component) > 0:
            dwave = np.roll(pp.lam, -1) - pp.lam
            dwave[-1] = dwave[-2] # fix the bad point introduced by roll
            # dwave = np.ones_like(pp.lam)
            # TODO: combined flux_scale into weights
            pp.gas_flux = dwave @ pp.matrix * pp.weights * pp.flux_scale
            pp.gas_flux_err = (dwave @ pp.matrix 
                    * capfit.cov_err(pp.matrix / pp.noise[:, None])[1] * pp.flux_scale)
            pp.gas_lines = dict(zip(pp.gas_names, pp.gas_flux))
            pp.gas_lines_err = dict(zip(pp.gas_names, pp.gas_flux_error))

        return pp
