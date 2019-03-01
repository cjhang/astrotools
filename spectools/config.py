"""Define the class for configuration, make it easier to access data from 
different projects
"""

import os
import configparser
import warnings
# from .exceptions import SpecToolsError

class ToolsConfig(object):
    """Basic configuration class for spectools
    """
    def __init__(self, config_file=None):
        '''Initiate the class

        Parameters
        ----------
        config_file : str or None
            specific the configuration file, None for the default
                         '~/.spectools'
        '''
        self.config = configparser.ConfigParser()
        if config_file is None:
            self.config_file = os.path.join(os.path.expanduser('~'), 
                                            '.spectools')
        elif os.path.isfile(config_file):
            self.config_file = config_file
        else:
            warnings.warn('Invalid configuration file!')
            self.config_file = None
        self.config.read(self.config_file)
        self.config_entries = list(self.config)


class SDSSConfig(ToolsConfig):
    """basic configuration class of Science Archive of SDSS
    """

    def __init__(self, config_file=None):
        """access the configuration for SDSS from file or environment variables
        """
        ToolsConfig.__init__(self, config_file=config_file)
        if 'sas' not in self.config_entries:
            self.config['sas'] = {}
        sas = self.config['sas']
        self.sas_base_dir = sas.get('base_dir', 
                                    os.path.expanduser('~') + '/SAS')
        # overwrite from the global variable if possible
        if os.getenv('SAS_BASE_DIR'):
            self.sas_base_dir = os.getenv('SAS_BASE_DIR')
        print("Global SAS directory is {0}".format(self.sas_base_dir))
        
        
class MaNGAConfig(SDSSConfig):
    """configuration class for MaNGA/SDSS
    """
    
    def __init__(self, config_file=None):
        """access the configuration for MaNGA from file or environment 
           variables
        """
        SDSSConfig.__init__(self, config_file=config_file)
        if 'manga' not in self.config_entries:
            self.config['manga'] = {}
        manga = self.config['manga']
        self.drp_version = manga.get('drp_version', 'v2_4_3')
        self.dap_version = manga.get('dap_version', '2.2.1')
        self.products = manga.get('products', 'HYB10')
        # overwrite by global variable if available
        if os.getenv('MANGA_DRP_VERSION'):
            self.drp_version = os.getenv('MANGA_DRP_VERSION')
        if os.getenv('MANGA_DAP_VERSION'):
            self.dap_version = os.getenv('MANGA_DAP_VERSION')
        if os.getenv('MANGA_PRODUCTS'):
            self.products = os.getenv('MANGA_PRODUCTS')
