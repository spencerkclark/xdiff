"""Options setting interface taken from xarray."""
EARTH_RADIUS = 6370997.0  # Radius of the Earth [m]

LON_DIM = 'lon_dim'
LAT_DIM = 'lat_dim'
PFULL_DIM = 'pfull_dim'
PHALF_DIM = 'phalf_dim'
TIME_DIM = 'time_dim'
RADIUS = 'radius'

OPTIONS = {
    LON_DIM: 'lon',
    LAT_DIM: 'lat',
    PFULL_DIM: 'pfull',
    PHALF_DIM: 'phalf',
    TIME_DIM: 'time',
    RADIUS: EARTH_RADIUS
}


def _positive_float(value):
    return isinstance(value, float) and value > 0


_VALIDATORS = {
    RADIUS: _positive_float,
}


class set_options(object):
    """Set options for xdiff in a controlled context.
    Currently supported options:
    - ``lon_dim``: name of longitude dimension for differentiation
      Default: ``'lon'``.
    - ``lat_dim``: name of latitude dimension for differentiation
      Default: ``'lat'``.
    - ``pfull_dim``: name of vertical level midpoints dimension
      Default: ``'pfull'``.
    - ``phalf_dim``: name of vertical level edges dimension
      Default: ``'phalf'``
    - ``time_dim``: name of time dimension for differentiation
      Default: ``'time'``
    - ``radius``: radius of the sphere used in computing derivatives
      Default: ``6370997.0``
    You can use ``set_options`` either as a context manager:
    >>> with xdiff.set_options(lon_dim='longitude', lat_dim='latitude'):
    ...     result = xdiff.d_dlon(da)
    Or to set global options:
    >>> xdiff.set_options(lon_dim='longitude', lat_dim='latitude')
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    'argument name %r is not in the set of valid options %r'
                    % (k, set(OPTIONS)))
            if k in _VALIDATORS and not _VALIDATORS[k](v):
                raise ValueError(
                    'option %r given an invalid value: %r' % (k, v))
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
