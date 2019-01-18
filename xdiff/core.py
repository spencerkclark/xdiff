"""Functions for computing finite differences on the sphere."""
import numpy as np
import xarray as xr

from .options import OPTIONS


def _option_or_default(**kwargs):
    options = []
    for key, value in kwargs.items():
        if value is None:
            options.append(OPTIONS[key])
        else:
            options.append(value)

    if len(options) == 1:
        return options[0]
    else:
        return tuple(options)


def add_cyclic_points_lon(da, extent=360., lon_dim=None):
    """Add cyclic points to both sides of a dimension.

    Assumes uneven grid spacing; uses strictly coordinates
    defined at the grid-box midpoints.

    Parameters
    ----------
    da : DataArray
        Input DataArray
    extent : float
        Longitudinal extent of the sphere (default 360.)
    lon_dim : str
        Longitude dimension of input DataArray

    Returns
    -------
    DataArray
    """
    lon_dim = _option_or_default(lon_dim=lon_dim)

    da_left = da.isel(**{lon_dim: -1})
    da_right = da.isel(**{lon_dim: 0})
    da_left[lon_dim] = da[lon_dim].isel(**{lon_dim: -1}) - extent
    da_right[lon_dim] = da[lon_dim].isel(**{lon_dim: 0}) + extent
    return xr.concat([da_left, da, da_right], dim=lon_dim)


def add_cyclic_points_lat(da, extent=180., lat_dim=None, lon_dim=None):
    """Adds cyclic points to both sides of a dimension

    Assumes uneven grid spacing; uses strictly coordinates
    defined at the grid-box midpoints.  Reflects the edge rows about
    the midpoint in longitude and appends them to the latitude edges.

    This is as in the finite volume advection scheme in the idealized moist
    model.

    For example, say you have a top row of values representing the last
    latitude circle in your model:

    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
    |   0   |   1   |   2   |   3   |   4   |   5   |
    |       |       |       |       |       |       |
    +-------+-------+-------+-------+-------+-------+

    This method adds a row on top that looks like:

    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
    |   3   |   4   |   5   |   0   |   1   |   2   |
    |       |       |       |       |       |       |
    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
    |   0   |   1   |   2   |   3   |   4   |   5   |
    |       |       |       |       |       |       |
    +-------+-------+-------+-------+-------+-------+

    Similarly, a row is added below the bottom latitude circle.

    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
    |   0   |   1   |   2   |   3   |   4   |   5   |
    |       |       |       |       |       |       |
    +-------+-------+-------+-------+-------+-------+
    |       |       |       |       |       |       |
    |   3   |   4   |   5   |   0   |   1   |   2   |
    |       |       |       |       |       |       |
    +-------+-------+-------+-------+-------+-------+

    This allows us to use centered differences in latitude.

    Parameters
    ----------
    da : DataArray
        Input DataArray
    extent : float
        Degrees in latitude
    lon_dim : str
        Longitude dimension or input DataArray
    lat_dim : str
        Latitude dimension of input DataArray

    Returns
    -------
    DataArray
    """
    lon_dim, lat_dim = _option_or_default(lon_dim=lon_dim, lat_dim=lat_dim)

    nlon = da.sizes[lon_dim]

    da_bottom = xr.concat(
        [da.isel(**{lat_dim: 0, lon_dim: slice(nlon // 2, None)}),
         da.isel(**{lat_dim: 0, lon_dim: slice(None, nlon // 2)})],
        dim=lon_dim)

    da_top = xr.concat(
        [da.isel(**{lat_dim: -1, lon_dim: slice(nlon // 2, None)}),
         da.isel(**{lat_dim: -1, lon_dim: slice(None, nlon // 2)})],
        dim=lon_dim)

    da_bottom[lon_dim] = da[lon_dim]
    da_top[lon_dim] = da[lon_dim]

    da_bottom[lat_dim] = -extent - da[lat_dim].isel(**{lat_dim: 0})
    da_top[lat_dim] = extent - da[lat_dim].isel(**{lat_dim: -1})

    return xr.concat([da_bottom, da, da_top], dim=lat_dim)


def d_dlon(arr, lon_dim=None, lat_dim=None, radius=None):
    """Differentiate an array with respect to longitude.

    Assumes a global domain, with periodic boundaries.  Uses second-order
    finite differencing with spherical metric factors.  Follows the methods of
    Seager and Henderson (2013)[1]_.

    Parameters
    ----------
    arr : DataArray
        Input DataArray
    lon_dim : str
        Longitude dimension of input DataArray
    lat_dim : str
        Latitude dimension of input DataArray
    radius : float
        Radius of the sphere

    Returns
    -------
    DataArray

    References
    ----------
    .. [1] Seager, R., & Henderson, N. (2013). Diagnostic Computation of
    Moisture Budgets in the ERA-Interim Reanalysis with Reference to Analysis
    of CMIP-Archived Atmospheric Model Data. Journal of Climate, 26(20),
    7876–7901. https://doi.org/10.1175/JCLI-D-13-00018.1
    """
    lon_dim, lat_dim, radius = _option_or_default(lon_dim=lon_dim,
                                                  lat_dim=lat_dim,
                                                  radius=radius)

    arr = add_cyclic_points_lon(arr)

    lon = np.deg2rad(arr[lon_dim])
    lat = np.deg2rad(arr[lat_dim])

    metric_factor = radius * np.cos(lat)

    dlon_left = lon - lon.shift(**{lon_dim: 1})
    dlon_right = lon.shift(**{lon_dim: -1}) - lon
    dlon_center = lon.shift(**{lon_dim: -1}) - lon.shift(**{lon_dim: 1})

    darr_left = arr - arr.shift(**{lon_dim: 1})
    darr_right = arr.shift(**{lon_dim: -1}) - arr

    result = (dlon_left * darr_right / dlon_right +
              dlon_right * darr_left / dlon_left) / dlon_center

    return result.isel(**{lon_dim: slice(1, -1)}) / metric_factor


def d_dlat(arr, divergence=False, lon_dim=None, lat_dim=None, radius=None):
    """Differentiate an array with respect to latitude.

    Assumes a global domain, with periodic boundaries.  Uses second-order
    finite differencing wtih spherical metric factors.  Follows the methods of
    Seager and Henderson (2013)[1]_.

    Parameters
    ----------
    arr : DataArray
        Input DataArray
    divergence : bool
        It True, treat the derivative as being the meridional term in the
        divergence of a vector field; if False, treat the derivative as the
        meridional derivative of a scalar field.
    lon_dim : str
        Longitude dimension of input DataArray
    lat_dim : str
        Latitude dimension of input DataArray
    radius : float
        Radius of the sphere

    Returns
    -------
    DataArray

    References
    ----------
    .. [1] Seager, R., & Henderson, N. (2013). Diagnostic Computation of
    Moisture Budgets in the ERA-Interim Reanalysis with Reference to Analysis
    of CMIP-Archived Atmospheric Model Data. Journal of Climate, 26(20),
    7876–7901. https://doi.org/10.1175/JCLI-D-13-00018.1
    """
    lon_dim, lat_dim, radius = _option_or_default(lon_dim=lon_dim,
                                                  lat_dim=lat_dim,
                                                  radius=radius)

    arr = add_cyclic_points_lat(arr)

    lat = np.deg2rad(arr[lat_dim])
    cos_lat = np.abs(np.cos(lat))  # The weighting is positive-definite

    if divergence:
        arr = cos_lat * arr
        metric_factor = radius * cos_lat
    else:
        metric_factor = radius

    dlat_left = lat - lat.shift(**{lat_dim: 1})
    dlat_right = lat.shift(**{lat_dim: -1}) - lat
    dlat_center = lat.shift(**{lat_dim: -1}) - lat.shift(**{lat_dim: 1})

    darr_left = arr - arr.shift(**{lat_dim: 1})
    darr_right = arr.shift(**{lat_dim: -1}) - arr

    result = (dlat_left * darr_right / dlat_right +
              dlat_right * darr_left / dlat_left) / dlat_center

    return result.isel(**{lat_dim: slice(1, -1)}) / metric_factor


def d_dt(da, time_dim=None, datetime_unit='s'):
    """Differentiate with respect to the time coordinate.

    Uses xarray's differentiate function which is based on numpy.gradient.

    Parameters
    ----------
    da : DataArray
        Input DataArray
    time_dim : str
        Dimension name of time coordinate
    datetime_unit : str
        Units for the time differences

    Returns
    -------
    DataArray
    """
    time_dim = _option_or_default(time_dim=time_dim)
    return da.differentiate(time_dim, datetime_unit=datetime_unit)


def replace_coord(arr, old_dim, new_dim, new_coord):
    """Replace a coordinate with new one; new and old must have same shape."""
    new_arr = arr.rename({old_dim: new_dim})
    ds = new_arr.to_dataset(name='new_arr')
    ds[new_dim] = new_coord
    return ds['new_arr']


def to_pfull_from_phalf(arr, pfull, pfull_dim=None, phalf_dim=None):
    """Compute data at full pressure levels from values at half levels."""
    pfull_dim, phalf_dim = _option_or_default(pfull_dim=pfull_dim,
                                              phalf_dim=phalf_dim)

    arr_top = arr.copy()[{phalf_dim: slice(1, None)}]
    arr_top = replace_coord(arr_top, phalf_dim, pfull_dim, pfull)

    arr_bot = arr.copy()[{phalf_dim: slice(None, -1)}]
    arr_bot = replace_coord(arr_bot, phalf_dim, pfull_dim, pfull)
    return 0.5 * (arr_bot + arr_top)


def d_dlon_const_p(arr, ps, pk, bk, lon_dim=None, lat_dim=None,
                   radius=None, pfull_dim=None, phalf_dim=None):
    """Take the gradient in longitude on surfaces of constant pressure.

    Assumes the model uses a hybrid vertical coordinate, where
    p = ak + bk * ps
    where p is the pressure at all vertical levels and ps is the surface
    pressure; ak and bk are the hybrid coordinate parameters.

    Follows from Equation B3 in Hill et al. (2017)[1]_.

    Parameters
    ----------
    arr : DataArray
        Input DataArray
    ps : DataArray
        Surface pressure DataArray
    pk : DataArray
        pk DataArray
    bk : DataArray
        bk DataArray
    lon_dim : str
        Longitude dimension
    lat_dim : str
        Latitude dimension
    radius : float
        Radius of the sphere
    pfull_dim : str
        Dimension name for level midpoints
    phalf_dim : str
        Dimension name for level edges

    Returns
    -------
    DataArray

    References
    ----------
    .. [1] Hill, S. A., Ming, Y., Held, I. M., & Zhao, M. (2017). A Moist
    Static Energy Budget–Based Analysis of the Sahel Rainfall Response to
    Uniform Oceanic Warming. Journal of Climate, 30(15),
    5637–5660. https://doi.org/10.1175/JCLI-D-16-0785.1
    """
    lon_dim, lat_dim, radius = _option_or_default(lon_dim=lon_dim,
                                                  lat_dim=lat_dim,
                                                  radius=radius)
    pfull_dim, phalf_dim = _option_or_default(pfull_dim=pfull_dim,
                                              phalf_dim=phalf_dim)

    vert = arr.differentiate(pfull_dim)
    bk_pfull = to_pfull_from_phalf(bk, arr[pfull_dim])
    ap = pk.diff(phalf_dim).drop(phalf_dim).rename({phalf_dim: pfull_dim})
    bp = bk.diff(phalf_dim).drop(phalf_dim).rename({phalf_dim: pfull_dim})

    term1 = d_dlon(arr, lon_dim=lon_dim, lat_dim=lat_dim, radius=radius)
    term2 = - vert * (bk_pfull / (ap + bp * ps)) * d_dlon(ps)
    return term1 + term2


def d_dlat_const_p(arr, ps, pk, bk, lon_dim=None, lat_dim=None,
                   radius=None, pfull_dim=None, phalf_dim=None):
    """Take the gradient in latitude on surfaces of constant pressure.

    Assumes the model uses a hybrid vertical coordinate, where
    p = ak + bk * ps
    where p is the pressure at all vertical levels and ps is the surface
    pressure; ak and bk are the hybrid coordinate parameters.

    Follows from Equation B3 in Hill et al. (2017)[1]_.

    Parameters
    ----------
    arr : DataArray
        Input DataArray
    ps : DataArray
        Surface pressure DataArray
    pk : DataArray
        pk DataArray
    bk : DataArray
        bk DataArray
    lon_dim : str
        Longitude dimension
    lat_dim : str
        Latitude dimension
    radius : float
        Radius of the sphere
    pfull_dim : str
        Dimension name for level midpoints
    phalf_dim : str
        Dimension name for level edges

    Returns
    -------
    DataArray

    References
    ----------
    .. [1] Hill, S. A., Ming, Y., Held, I. M., & Zhao, M. (2017). A Moist
    Static Energy Budget–Based Analysis of the Sahel Rainfall Response to
    Uniform Oceanic Warming. Journal of Climate, 30(15),
    5637–5660. https://doi.org/10.1175/JCLI-D-16-0785.1
    """
    lon_dim, lat_dim, radius = _option_or_default(lon_dim=lon_dim,
                                                  lat_dim=lat_dim,
                                                  radius=radius)
    pfull_dim, phalf_dim = _option_or_default(pfull_dim=pfull_dim,
                                              phalf_dim=phalf_dim)

    vert = arr.differentiate(pfull_dim)
    bk_pfull = to_pfull_from_phalf(bk, arr[pfull_dim])
    ap = pk.diff(phalf_dim).drop(phalf_dim).rename({phalf_dim: pfull_dim})
    bp = bk.diff(phalf_dim).drop(phalf_dim).rename({phalf_dim: pfull_dim})

    term1 = d_dlat(arr, lon_dim=lon_dim, lat_dim=lat_dim, radius=radius)
    term2 = - vert * (bk_pfull / (ap + bp * ps)) * d_dlat(ps)
    return term1 + term2
