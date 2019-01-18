import numpy as np
import pytest
import xarray as xr

from xdiff import d_dlon, d_dlat, EARTH_RADIUS, set_options


def f(lon, lat, wavenumber):
    rad_lon = np.deg2rad(lon)
    rad_lat = np.deg2rad(lat)
    return EARTH_RADIUS * np.cos(rad_lat) * np.sin(wavenumber * rad_lon)


@pytest.fixture(params=['lon', 'longitude'])
def lon_name(request):
    return request.param


@pytest.fixture(params=['lat', 'latitude'])
def lat_name(request):
    return request.param


@pytest.fixture()
def lon(lon_name):
    DLON = 1.
    lon = np.arange(0. + DLON / 2., 360., DLON)
    return xr.DataArray(lon, [(lon_name, lon)])


@pytest.fixture()
def lat(lat_name):
    DLAT = 1.
    lat = np.arange(-90. + DLAT / 2., 90., DLAT)
    return xr.DataArray(lat, [(lat_name, lat)])


@pytest.fixture()
def da_wavenumber_5(lon, lat):
    return f(lon, lat, 5)


@pytest.fixture()
def da_wavenumber_6(lon, lat):
    return f(lon, lat, 6)


def test_d_dlon(lon, lat, da_wavenumber_5, lon_name, lat_name):
    rad_lon = np.deg2rad(lon)
    wavenumber = 5
    expected = wavenumber * np.cos(wavenumber * rad_lon)
    expected, _ = xr.broadcast(expected, da_wavenumber_5)
    with set_options(lon_dim=lon_name, lat_dim=lat_name):
        result = d_dlon(da_wavenumber_5)
    xr.testing.assert_allclose(result, expected, rtol=0., atol=0.1)


def test_d_dlat_gradient(lon, lat, da_wavenumber_5, lon_name, lat_name):
    rad_lon = np.deg2rad(lon)
    rad_lat = np.deg2rad(lat)
    wavenumber = 5
    expected = - np.sin(rad_lat) * np.sin(wavenumber * rad_lon)
    with set_options(lon_dim=lon_name, lat_dim=lat_name):
        result = d_dlat(da_wavenumber_5, divergence=False)
    xr.testing.assert_allclose(result, expected, rtol=0., atol=0.1)


def test_d_dlat_divgerence(lon, lat, da_wavenumber_6, lon_name, lat_name):
    rad_lon = np.deg2rad(lon)
    rad_lat = np.deg2rad(lat)
    wavenumber = 6
    expected = - 2 * np.sin(rad_lat) * np.sin(wavenumber * rad_lon)
    with set_options(lon_dim=lon_name, lat_dim=lat_name):
        result = d_dlat(da_wavenumber_6, divergence=True)
    xr.testing.assert_allclose(result, expected, rtol=0., atol=0.1)
