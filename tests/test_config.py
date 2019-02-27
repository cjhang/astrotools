import os
import pytest
import configparser

from spectools.config import SDSSConfig, MaNGAConfig

@pytest.fixture()
def config_path(tmpdir):
    yield tmpdir.mkdir

@pytest.fixture()
def config_file(tmpdir):
    tmp_config = tmpdir.mkdir('spectools').join('.spectools')
    yield tmp_config

@pytest.fixture()
def goodconfig(config_file):
    config = configparser.ConfigParser()
    config['sas'] = {'base_dir': '/tmp/SAS'}
    config['manga'] = {}
    manga = config['manga']
    manga['drp_version'] = 'v1_0_0'
    manga['dap_version'] = '1.0.0'
    # config.write(config_file)
    with open(config_file, 'w') as configfile:
       config.write(configfile)
    return config_file

@pytest.fixture()
def badconfig(config_file):
    config = configparser.ConfigParser()
    config['sas2'] = {'base_dir': '/tmp/SAS'}
    with open(config_file, 'w') as configfile:
       config.write(configfile)
    return config_file

def test_sdss_goodconfig(goodconfig):
    sdss_config = SDSSConfig(config_file=goodconfig)
    assert sdss_config.sas_base_dir == '/tmp/SAS'

def test_sdss_badconfig(badconfig):
    sdss_config = SDSSConfig(config_file=badconfig)
    assert sdss_config.sas_base_dir == os.path.join(os.path.expanduser('~'), 
                                                    'SAS')

def test_manga(goodconfig):
    manga_config = MaNGAConfig(config_file=goodconfig)
    assert manga_config.drp_version == 'v1_0_0'
    assert manga_config.dap_version == '1.0.0'
