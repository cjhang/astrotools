import os
import pytest

import numpy as np
from astropy import io
import spectools as spectools_package
from spectools.spectrum import Spectrum
from spectools.datacube import Datacube

@pytest.fixture()
def config_path(tmpdir):
    yield tmpdir.mkdir

def test_datacube():
    dc = Datacube()
    assert dc.flux == None
    assert dc.z == None

def test_fitting(tmpdir):
    dc = Datacube()
    # read the spectrum data
    package_path = os.path.dirname(os.path.realpath(spectools_package.__file__))
    fpath = package_path + '/data/spectra/manga-8141-1901-LOGCUBE-HYB10-GAU-MILESHC.fits.gz'
    with io.fits.open(fpath) as f:
        dc.wave = f['wave'].data
        dc.flux = f['flux'].data
        dc.noise = 1 / np.sqrt(f['ivar'].data + 1e-8) 
        # get central spaxel
        sp = dc[17, 17]
        assert isinstance(sp, Spectrum)
        assert sp.z == dc.z
    dc.z = 0.0312591
    dc.instru_sigma = 2.62
    pix, naxis1, naxis2 = dc.flux.shape
    # fitting the central four pixels
    mask = np.full((naxis1, naxis2), True)
    mask[naxis1//2:naxis1//2+1, naxis2//2:naxis2//2+1] = False
    fmap = dc.fitmap('test', directory=tmpdir, quiet=True, mask=mask, 
                     save_map=True, save_cube=True)
    
    assert os.path.isfile(os.path.join(tmpdir, 'test.fits'))
    assert os.path.isfile(os.path.join(tmpdir, 'test-cubes.fits'))
