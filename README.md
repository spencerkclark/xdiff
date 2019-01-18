xdiff
=====

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
