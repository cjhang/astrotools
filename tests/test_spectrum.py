import os
import pytest

import numpy as np
from astropy import io
import spectools as spectools_package
from spectools.spectrum import Spectrum

def test_spectrum():
    sp = Spectrum()
    assert sp.flux == None
    assert sp.z == None
    assert sp.fitted == False

def test_spectrum_ppxf_kinematics():
    sp = Spectrum()
    # read the spectrum data
    package_path = os.path.dirname(os.path.realpath(spectools_package.__file__))
    fpath = package_path + '/data/spectra/NGC4636_SDSS_DR12.fits'
    with io.fits.open(fpath) as f:
        data = f['coadd'].data
        sp.wave = 10**data['loglam']
        sp.flux = data['flux']
        sp.noise = np.full_like(data['flux'], 0.0166)
        sp.instru_sigma = 2.7
        sp.z = 0.003129

        pp = sp.ppxf_fit(mode='kinematics', quiet=True)
        assert sp.fitted == True
        assert pp.chi2 is not None
