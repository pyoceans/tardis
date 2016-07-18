from __future__ import division, absolute_import

import numpy as np

import iris
from iris import Constraint

from .coords import z_coord, t_coord

iris.FUTURE.netcdf_promote = True
iris.FUTURE.cell_datetime_objects = True


__all__ = ['find_surface',
           'find_time',
           'find_bbox',
           'extract_surface',
           'extract_time',
           'extract_bbox']


def find_surface(cube):
    """
    Return the `cube` index for the surface layer of for any model grid
    (rgrid, ugrid, sgrid), and any non-dimensional coordinate.

    TODO: Fold this into `find_layer()`

    """
    z = z_coord(cube)
    if not z:
        msg = "Cannot find the surface for cube {!r}".format
        raise ValueError(msg(cube))
    else:
        if np.argmin(z.shape) == 0 and z.ndim == 2:
            points = z[:, 0].points
        elif np.argmin(z.shape) == 1 and z.ndim == 2:
            points = z[0, :].points
        else:
            points = z.points
        positive = z.attributes.get('positive', None)
        if positive == 'up':
            idx = np.unique(points.argmax(axis=0))[0]
        else:
            idx = np.unique(points.argmin(axis=0))[0]
        return idx


def find_time(cube, datetime):
    """
    Return the `cube` nearest index to a `datetime` object.

    Examples
    --------
    >>> import iris
    >>> import numpy as np
    >>> from datetime import datetime
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/"
    ...        "SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> isinstance(find_time(cube, datetime.utcnow()), np.integer)
    True

    """
    timevar = t_coord(cube)
    try:
        time = timevar.units.date2num(datetime)
        idx = timevar.nearest_neighbour_index(time)
    except IndexError:
        idx = -1
    return idx


def _minmax(v):
    return np.min(v), np.max(v)


def find_bbox(cube, bbox):
    """
    Get the four corner indices of a `cube` given a `bbox`.

    Examples
    --------
    >>> import iris
    >>> import numpy as np
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/"
    ...        "SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> bbox = [-87.40, -74.70, 24.25, 36.70]
    >>> idxs = find_bbox(cube, bbox)
    >>> [isinstance(idx, np.integer) for idx in idxs]
    [True, True, True, True]

    """
    from oceans import wrap_lon180
    lons = cube.coords(axis='X')[0].points
    lats = cube.coords(axis='Y')[0].points
    lons = wrap_lon180(lons)

    inregion = np.logical_and(np.logical_and(lons > bbox[0],
                                             lons < bbox[1]),
                              np.logical_and(lats > bbox[2],
                                             lats < bbox[3]))
    region_inds = np.where(inregion)
    imin, imax = _minmax(region_inds[0])
    jmin, jmax = _minmax(region_inds[1])
    return imin, imax+1, jmin, jmax+1


def extract_surface(cube):
    """
    Extract the `cube` surface layer using `find_surface`.
    This is a work around `iris.cube.Cube.slices` error:
        'The requested coordinates are not orthogonal.'

    Examples
    --------
    >>> import iris
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/"
    ...        "SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> cube.ndim == 4
    True
    >>> extract_surface(cube).ndim == 3
    True

    """
    idx = find_surface(cube)
    if cube.ndim == 4:
        return cube[:, int(idx), ...]
    elif cube.ndim == 3:
        return cube[int(idx), ...]
    else:
        msg = "Cannot find the surface for cube {!r}".format
        raise ValueError(msg(cube))


def extract_time(cube, start, stop=None):
    """
    Slice time by indexes using a nearest criteria.
    NOTE: Assumes time is the first dimension!

    Examples
    --------
    >>> import iris
    >>> from datetime import datetime, timedelta
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/"
    ...        "SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> stop = datetime.utcnow()
    >>> start = stop - timedelta(days=7)
    >>> extract_time(cube, start, stop).shape[0] < cube.shape[0]
    True

    """
    istart = find_time(cube, start)
    if stop:
        istop = find_time(cube, stop)
        if istart == istop:
            raise ValueError('istart must be different from istop! '
                             'Got istart {!r} and '
                             ' istop {!r}'.format(istart, istop))
        return cube[istart:istop, ...]
    else:
        return cube[istart, ...]


def extract_bbox(cube, bbox, grid_type):
    """
    Extract a subset of a cube inside a lon, lat bounding box (`bbox`)
    bbox = [lon_min lon_max lat_min lat_max].

    NOTE: This is a work around too subset an iris cube that has
    2D lon, lat coords.

    Examples
    --------
    >>> import iris
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/"
    ...        "SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> bbox = [-87.40, -74.70, 24.25, 36.70]
    >>> c = extract_bbox(cube, bbox, grid_type='2D_curvilinear')
    >>> c.shape < cube.shape
    True

    """
    if grid_type == 'ugrid':
        lat = Constraint(latitude=lambda cell: bbox[1] <= cell <= bbox[3])
        lon = Constraint(longitude=lambda cell: bbox[0] <= cell <= bbox[2])
        cube = cube.extract(lon & lat)
    elif grid_type == 'sgrid' or grid_type == '2D_curvilinear':
        imin, imax, jmin, jmax = find_bbox(cube, bbox)
        if cube.ndim > 2:
            cube = cube[..., imin:imax, jmin:jmax]
        elif cube.ndim == 2:
            cube = cube[imin:imax, jmin:jmax]
        else:
            msg = 'Cannot subset {!r} with bbox {}'.format
            raise ValueError(msg(cube, bbox))

    # NOTE: `rgrid` and `unknown` are passed to iris`.intersection()`
    # intersection should deal 0-360 properly.
    else:
        cube = cube.intersection(longitude=(bbox[0], bbox[2]),
                                 latitude=(bbox[1], bbox[3]))
    return cube


if __name__ == '__main__':
    import doctest
    doctest.testmod()
