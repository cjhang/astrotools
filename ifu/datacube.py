"""This is the core class that handle datacubes from IFU observation, including
function to do spectrum fitting of the whole cube and generate the results into
files

:TODO: add a visual tools to handle all the manipulation

"""

from spectools.spectrtrum import Spectrum


class Datacube(object):
    """ The basic class to handle datacube
    """

    def __init__(self):
        pass

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
        if ra is not None or dec is not None:
            sp_coor = coordinates.SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
            radec_xy = wcs.utils.skycoord_to_pixel(sp_coor, self.wcs_cude)
            x = np.round(radec_xy[1]).astype(int)
            y = np.round(radec_xy[0]).astype(int)
        spaxel.flux = self.flux[:, x, y]
        spaxel.redcorr = self.redcorr
        spaxel.noise = self.noise[:, x, y]
        spaxel.load_fit = self.load_fit
        if self.load_fit:
            spaxel.model = self.model[:, x, y]
            spaxel.stellarcontinuum = self.stellarcontinuum[:, x, y]
            spaxel.emlines = self.emlines[:, x, y]
            spaxel.emline_base = self.emline_base[:, x, y]
            spaxel.residual = self.residual[:, x, y]
        return spaxel


