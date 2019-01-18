import pytest

import xdiff
from xdiff.options import OPTIONS


def test_invalid_option_raises():
    with pytest.raises(ValueError):
        xdiff.set_options(not_a_valid_options=True)


def test_lon_dim():
    assert OPTIONS['lon_dim'] == 'lon'
    with xdiff.set_options(lon_dim='longitude'):
        assert OPTIONS['lon_dim'] == 'longitude'


def test_lat_dim():
    assert OPTIONS['lat_dim'] == 'lat'
    with xdiff.set_options(lat_dim='latitude'):
        assert OPTIONS['lat_dim'] == 'latitude'


def test_pfull_dim():
    assert OPTIONS['pfull_dim'] == 'pfull'
    with xdiff.set_options(pfull_dim='pf'):
        assert OPTIONS['pfull_dim'] == 'pf'


def test_phalf_dim():
    assert OPTIONS['phalf_dim'] == 'phalf'
    with xdiff.set_options(phalf_dim='ph'):
        assert OPTIONS['phalf_dim'] == 'ph'


def test_time_dim():
    assert OPTIONS['time_dim'] == 'time'
    with xdiff.set_options(time_dim='t'):
        assert OPTIONS['time_dim'] == 't'


def test_radius():
    assert OPTIONS['radius'] == 6370997.0
    with xdiff.set_options(radius=5.):
        assert OPTIONS['radius'] == 5.
    with pytest.raises(ValueError):
        with xdiff.set_options(radius='a'):
            pass
