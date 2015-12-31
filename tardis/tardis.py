import numpy as np


import iris
from iris.cube import CubeList

import pyugrid
import pysgrid  # NOTE: Really?! How many custom exceptions to say ValueError?
from pysgrid.custom_exceptions import SGridNonCompliantError

from .utils import wrap_lon180, cf_name_list

__all__ = ['load_phenomena',
           'OceanModelCube',
           'cf_name_list']


def _get_grid(self):
    grid = 'unknown'
    # SGRID.
    try:
        return pysgrid.from_ncfile(self.filename)
    except SGridNonCompliantError:
        pass
    # UGRID.
    try:
        return pyugrid.UGrid.from_ncfile(self.filename)
    except ValueError:
        pass
    return grid


def _get_grid_type(self):
    xdim = self.cube.coord(axis='X').ndim
    ydim = self.cube.coord(axis='Y').ndim
    if isinstance(self.grid, pysgrid.SGrid2D):
        grid_type = 'sgrid'
    elif isinstance(self.grid, pyugrid.UGrid):
        grid_type = 'ugrid'
    elif xdim == 1 and ydim == 1:
        grid_type = 'rgrid'
    elif xdim == 2 and ydim == 2:
        grid_type = '2D_curvilinear'
    else:
        grid_type = 'unknown'
    return grid_type


def _filter_none(lista):
    return [x for x in lista if x is not None]


def _in_list(cube, name_list):
    return cube.standard_name in name_list


def load_phenomena(url, name_list, callback=None, strict=False):
    """
    Return cube(s) for a certain phenomena in `name_list`.
    The `name_list` must be a collection of CF-1.6 `standard_name`s.

    If `strict` is set to True the function will return **only** one cube,
    if only one is expected to exist, otherwise an exception will be raise.
    (Similar to iris `extract_strict` method.)

    The user may also pass a `callback` function to coerce the metadata
    to CF-conventions.

    Examples
    --------
    >>> import iris
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/sabgom/"
    ...        "SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> name_list = cf_name_list['sea_water_temperature']
    >>> cubes = load_phenomena(url, name_list)
    >>> cube = load_phenomena(url, name_list, strict=True)
    >>> isinstance(cubes, CubeList)
    True
    >>> isinstance(cube, iris.cube.Cube)
    True
    """

    cubes = iris.load_raw(url, callback=callback)
    cubes = [cube for cube in cubes if _in_list(cube, name_list)]
    cubes = _filter_none(cubes)
    cubes = CubeList(cubes)
    if not cubes:
        raise ValueError('Cannot find {!r} in {}.'.format(name_list, url))
    if strict:
        if len(cubes) == 1:
            return cubes[0]
        else:
            msg = "> 1 cube found!  Expected just one.\n {!r}".format
        raise ValueError(msg(cubes))
    return cubes


class OceanModelCube(object):
    """
    Simple class that contains the cube and some extra goodies.

    """

    def __init__(self, cube, filename=None):
        if isinstance(cube, iris.cube.Cube):
            self.cube = cube
        else:
            msg = "Expected an iris cube.  Got {!r}.".format
            raise ValueError(msg(cube))

        self._kdtree = None
        self.filename = filename
        self.grid = _get_grid(self)
        self.grid_type = _get_grid_type(self)
        # NOTE: I always wrap longitude between -180, 180.
        self.lon = wrap_lon180(cube.coords(axis='X')[0].points)
        self.lat = cube.coords(axis='Y')[0].points

    def __repr__(self):
        msg = "<OceanModelCube of {}. Grid type: {}>".format
        return msg(self.cube.summary(shorten=True, name_padding=1),
                   self.grid_type)

    def get_nearest_grid_points(self, xi, yi, k=1):
        """
        Find `k` nearest model grid points from an iris `cube` at station
        lon: `xi`, lat: `yi`.  Returns the distance and indices to the grid
        input grid point.

        """
        if self._kdtree is None:
            self._make_tree()

        distances, indices = self._kdtree.query(np.array([xi, yi]).T, k=k)

        if self.grid_type == 'rgrid':
            shape = (self.lat.shape[0], self.lon.shape[0])
        else:
            assert (self.lon.shape == self.lat.shape)
            shape = self.lon.shape
        idx = np.unravel_index(indices, shape)
        return distances, idx

    def _make_tree(self):
        from scipy.spatial import cKDTree as KDTree
        lon, lat = self.lon, self.lat
        if self.grid_type == 'rgrid':
            lon, lat = np.meshgrid(lon, lat)
        # 2D_curvilinear, sgrid and ugrid are already paired.
        self._kdtree = KDTree(list(zip(lon.ravel(), lat.ravel())))

if __name__ == '__main__':
    import doctest
    doctest.testmod()
