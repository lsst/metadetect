"""
test configs

note we allow the get_all_metacal to do its own verifiction
of the metacal sub config
"""
import pytest

from metadetect.lsst.configs import get_config
from metadetect.lsst.defaults import DEFAULT_FWHM_SMOOTH


def test_configs_smoke():
    config = get_config()

    # make sure the default is verified
    get_config(config)

    with pytest.raises(ValueError):
        get_config({'blah': 3})


def test_weight_config():
    # make sure the default is verified
    get_config()
    inconfig = {}
    get_config(inconfig)

    fwhm = 1.2
    fwhm_smooth = 0.8
    for wtc in [{'fwhm': fwhm}, {'fwhm': fwhm, 'fwhm_smooth': fwhm_smooth}]:

        inconfig = {'weight': wtc}
        config = get_config(inconfig)
        assert config['weight']['fwhm'] == fwhm

        for key in wtc:
            assert config['weight'][key] == wtc[key]

        if 'fwhm_smooth' not in wtc:
            assert config['weight']['fwhm_smooth'] == DEFAULT_FWHM_SMOOTH

    with pytest.raises(ValueError):
        get_config({'weight': {'blah': 3}})


def test_detect_config():
    in_config = {'detect': {'thresh': 5}}
    config = get_config()
    assert config['detect']['thresh'] == in_config['detect']['thresh']

    with pytest.raises(ValueError):
        get_config({'detect': {'blah': 5}})
