xdiff
=====

[![Build Status](https://travis-ci.org/spencerkclark/xdiff.svg?branch=master)](https://travis-ci.org/spencerkclark/xdiff)

Generic differentiation routines on the sphere.  These make no assumptions
about the grid type (e.g. Arakawa C, etc.) or advection scheme used.  Instead
these are basic second-order differences, assuming grid coordinates are at
the grid cell centers, but on the sphere.  It is assumed the boundaries in
latitude and longitude are periodic, i.e. that things are on a global domain.
Therefore, centered differences are possible everywhere.

`xdiff` supports taking the derivative of both scalar (e.g. when computing a
gradient) and vector components (e.g. when computing a divergence).  Be sure to
use the appropriate option when taking a meridional derivative, see [Del in
cylindrical and spherical
coordinates](https://en.wikipedia.org/wiki/Del_in_cylindrical_and_spherical_coordinates). In
the zonal direction, there is no distinction.

Usage
-----

`xdiff` supports taking zonal and meridional derivatives on global data; data
must be global, because at the moment, periodicity in both dimensions is
assumed.  

```python
import numpy as np
import xarray as xr

import xdiff

DLON = 1.
lon = np.arange(0. + DLON / 2., 360., DLON)
lon = xr.DataArray(lon, [('lon', lon)])

DLAT = 1.
lat = np.arange(-90. + DLAT / 2., 90., DLAT)
lat = xr.DataArray(lat, [('lat', lat)])

rad_lon = np.deg2rad(lon)
rad_lat = np.deg2rad(lat)
f = xdiff.EARTH_RADIUS * np.cos(rad_lat) * np.sin(6 * rad_lon)

df_dlon = xdiff.d_dlon(f)
df_dlat = xdiff.d_dlat(f)
```

Note that if the names for longitude and latitude were not the default ('lon'
and 'lat'), we could specify those dimension names into the call to `d_dlon` or
`d_dlat`: 
```python
df_dlon = xdiff.d_dlon(f, lon_dim='longitude', lat_dim='latitude')
df_dlat = xdiff.d_dlat(f, lon_dim='longitude', lat_dim='latitude')
```

If you found yourself doing this a lot, you could reset the global default
options:
```python
xdiff.set_options(lon_dim='longitude', lat_dim='latitude')
df_dlon = d_dlon(f)
df_dlat = d_dlat(f)
```

Installation
------------

Currently the only option for installing `xdiff` is from source:
```
$ git clone https://github.com/spencerkclark/xdiff.git
$ cd xdiff
$ pip install -e .
```

References
----------

Hill, S. A., Ming, Y., Held, I. M., & Zhao, M. (2017). A Moist Static Energy
Budget–Based Analysis of the Sahel Rainfall Response to Uniform Oceanic
Warming. Journal of Climate, 30(15),
5637–5660. https://doi.org/10.1175/JCLI-D-16-0785.1

Seager, R., & Henderson, N. (2013). Diagnostic Computation of Moisture Budgets
in the ERA-Interim Reanalysis with Reference to Analysis of CMIP-Archived
Atmospheric Model Data. Journal of Climate, 26(20),
7876–7901. https://doi.org/10.1175/JCLI-D-13-00018.1
