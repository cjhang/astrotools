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
from  .utils import ppxf, miles_util, ppxf_util, capfit
from spectools.exceptions import SpecToolsError

class Spectrum(object):
    """The basic class for spectrum
    """

    def __init__(self, wave=None, z=None, flux=None, noise=None, 
                 instru_sigma=None, redcorr=None, flag=None):
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
        self.wave = np.asarray(wave)
        self.z = z
        self.wave_rest = self.wave/(1+z)
        self.flux = np.asarray(flux)
        self.noise = np.asarray(noise)
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
        self.flag = flag
        flux_invalid = np.ma.masked_invalid(self.flux)
        noise_invalid = np.ma.masked_invalid(self.noise)
        if flag is not None:
            self.flag = flag | flux_invalid.mask | noise_invalid.mask
        else:
            self.flag = flux_invalid.mask | noise_invalid.mask
        self.flux = flux_invalid.filled(0.0)
        self.noise = noise_invalid.filled(1e-4)
        #self.sigma = self.psf/2.355

    def read(self, filename):
        '''read from fits file with standard header
        
        Params:
            filename:
        '''
        pass

    def plot(self, waveRange=None,
             showFlux=True, showNoise=True, ax=None,
             showModel=False, showEmlines=False, showContinuum=False,
             showResidual=False, mask=None):
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
        mask : ndarray
            the good pixels included
        """
        if not ax:
            fig = plt.figure(figsize=(6,5))
            ax = fig.add_subplot(111)
        if showFlux:
            ax.step(self.wave, self.flux, label='flux', color='C0', lw=1)
        if showNoise and self.noise is not None:
            ax.fill_between(self.wave, self.flux-self.noise, self.flux+self.noise, color='red', alpha=0.4, step='pre')
        if self.flag is not None:
            masked_region = np.where(self.flag)
            for w in masked_region[0]:
                ax.axvspan(self.wave[w-1], self.wave[w], facecolor='lightgray')


        if self.fitted:
            if showModel:
                ax.plot(self.wave_fit, self.model, label='model', color='C1', lw=1)
            if showEmlines:
                ax.plot(self.wave_fit, self.emlines, label='emlines', color='C2', lw=1)
            if showContinuum:
                ax.plot(self.wave_fit, self.stellarcontinuum, label='stellar continuum',
                        color='C3', lw=1)
            if showResidual:
                ax.step(self.wave_fit, self.residual-0.5, label='residual', 
                        where='mid', color='0.5', lw=0.5)
        if waveRange:
            wave_window = (wave > waveRange[0]) & (wave < waveRange[1]) 
            ax.set_xlim(waveRange)
            ax.set_ylim(self.flux[wave_window].min(), 
                        self.flux[wave_window].max())
        ax.set_xlabel('restframe wavelength ($\AA$)')
        # ax.set_ylabel('flux ($10^{-20} erg\,s^{-1}\,cm^2\,\AA$)')
        ax.set_ylabel('Flux')
        ax.legend()

        if not ax:
            return fig

    def ppxf_fit(self, wave_window=None, 
                 tie_balmer=False, limit_doublets=False, quiet=False, 
                 mode='population', broad_balmer=None, broad_O3=None, 
                 fewer_lines=False, **kwargs):
        """fitting the spectrum using ppxf
        
        Parameters
        ----------
        wave_window : array_like
            Wavelength window to be fitted, usually determined by the template
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
        fit_window = wave_window #& (~self.flag)

        wave = self.wave_rest[fit_window]
        flux_scale = max(np.ma.median(np.ma.masked_invalid(self.flux[fit_window])),  1.)
        flux = self.flux[fit_window]/flux_scale
        if self.redcorr is not None:
            flux = flux * self.redcorr[fit_window]
        noise = self.noise[fit_window]/flux_scale

        # logrebin the spectrum
        flux_rebin, logLam1, velscale = ppxf_util.log_rebin([wave[0], wave[-1]], flux, oversample=1, flux=False)
        wave_rebin = np.exp(logLam1)
        noise_rebin2, logLam1, velscale = ppxf_util.log_rebin([wave[0], wave[-1]], noise**2)
        noise_rebin =  np.sqrt(noise_rebin2)

        self.wave_rebin = wave_rebin
        self.flux_rebin = flux_rebin

        FWHM_gal = self.instru_sigma
        
        # set the stellar templates
        package_path = os.path.dirname(os.path.realpath(spectools.__file__))
        miles = miles_util.miles(package_path + '/data/miles_models/Mun1.30*.fits', 
                                 velscale, FWHM_gal)
        reg_dim = miles.templates.shape[1:]
        stars_templates = miles.templates.reshape(miles.templates.shape[0], -1)
        

        # set the emission line templates
        lam_range_gal = np.array([np.min(wave), np.max(wave)])/(1 + self.z)
        gas_templates, gas_names, line_wave = ppxf_util.emission_lines(
                                                miles.log_lam_temp, lam_range_gal, FWHM_gal,
                                                tie_balmer=tie_balmer, limit_doublets=limit_doublets)



        templates = np.column_stack([stars_templates, gas_templates])
        

        # before fitting
        c = 299792.458
        dv = c*(miles.log_lam_temp[0] - np.log(wave[0]))
        # vel = c*np.log(1+self.z)
        start = [0, 180.]

        n_temps = stars_templates.shape[1]
        n_forbidden = np.sum(["[" in a for a in gas_names])  # forbidden lines contain "[*]"
        n_balmer = len(gas_names) - n_forbidden

        component = [0]*n_temps + [1]*n_balmer + [2]*n_forbidden
        gas_component = np.array(component) > 0  # gas_component=True for gas templates

        moments = [4, 2, 2]

        # Adopt the same starting value for the stars and the two gas components
        start = [start, start, start]
        
        mask = ~(self.flag[fit_window])
        muse_mask = ~((np.exp(logLam1) < 6020/(1+self.z)) & (np.exp(logLam1) > 5790/(1+self.z)))

        pp = ppxf.ppxf(templates, flux_rebin, noise_rebin, velscale, start,
              moments=moments, degree=-1, mdegree=10, vsyst=dv,
              lam=wave_rebin, clean=False, regul=0, reg_dim=reg_dim,
              component=component, gas_component=gas_component,
              gas_names=gas_names, gas_reddening=True,
              mask=mask, quiet=quiet, **kwargs)

        pp.flux_scale = flux_scale

        dwave = np.roll(pp.lam, -1) - pp.lam
        dwave[-1] = dwave[-2] # fix the bad point introduced by roll
        # self.model = dwave @ pp.matrix * pp.weights * flux_scale
        # self.model_err = flux_err = dwave @ pp.matrix * capfit.cov_err(pp.matrix / pp.noise[:, None])[1] * flux_scale
        self.wave_fit = pp.lam
        self.model = pp.bestfit * flux_scale
        self.fitted = True
    
        if not quiet:
            print('Desired Delta Chi^2: %.4g' % np.sqrt(2*flux_rebin.size))
            print('Current Delta Chi^2: %.4g' % ((pp.chi2 - 1)*flux_rebin.size))

            weights = pp.weights[~gas_component]  # Exclude weights of the gas templates
            weights = weights.reshape(reg_dim)/weights.sum()  # Normalized

            miles.mean_age_metal(weights)
            # miles.mass_to_light(weights, band="r")

        return pp

