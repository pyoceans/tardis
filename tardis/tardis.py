import numpy as np


import iris

import pyugrid
import pysgrid  # NOTE: Really?! How many custom exceptions to say ValueError?
from pysgrid.custom_exceptions import SGridNonCompliantError

from .coords import x_coord, y_coord
from .utils import wrap_lon180


"""
TODO:
- pyresample for 2D Coords re-gridding.
  http://pyresample.readthedocs.org/en/latest/installation.html
- pykdtree as alternative to SciPy's KDTree.
  https://github.com/storpipfugl/pykdtree
- pysgrid
  - From cube (and drop filename/URL).
  - Expose some methods.
    http://nbviewer.ipython.org/github/sgrid/pysgrid/blob/master/pysgrid/notebook_examples/hudson_shelf_valley.ipynb
- pyugrid
  - from cube (and drop filename/URL).
  - expose some methods.
  - https://ocefpaf.github.io/python4oceanographers/blog/2015/07/20/pyugrid
- matplotlib.tri interpolation.
  - http://matplotlib.org/examples/pylab_examples/triinterp_demo.html
- Cartesian KDTree.
  https://github.com/SciTools/iris/blob/00a168f5260c7ac765e0e307655d88cc218b2799/lib/iris/analysis/interpolate.py
- z_slices
  - How would cube.slices(['latitude', 'longitude']).next() work here?
- Use sos_names and look for all standard_names.  This is dangerous, but very
  convenient.
- trajectory

"""


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
        self.lon = wrap_lon180(x_coord(cube).points)
        self.lat = y_coord(cube).points

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
    import warnings

    def load_ecube(url, standard_name='sea_water_potential_temperature'):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cube = iris.load_cube(url, standard_name)
        return OceanModelCube(cube, filename=url)

    def verbose(ecube):
        print("{!r}".format(ecube))
        print("Grid object: {}".format(ecube.grid))

    bbox = (272.6, 24.25, 285.3, 36.70)

    print("\nRGRID and Z-coords\n")
    url = "http://oos.soest.hawaii.edu/thredds/dodsC/pacioos/hycom/global"
    rgrid = load_ecube(url)

    verbose(rgrid)

    if False:
        subset = rgrid.subset(bbox)
        print("Sub-setting with bbox {}.\n"
              "Original: {}\nnew: {}".format(bbox, rgrid.cube.shape,
                                             subset.shape))

    print("\nNon-compliant SGRID and Non-dimension Z-coords.\n")
    url = ("http://tds.marine.rutgers.edu/"
           "thredds/dodsC/roms/espresso/2013_da/his_Best/"
           "ESPRESSO_Real-Time_v2_History_Best_Available_best.ncd")

    curv = load_ecube(url)
    verbose(curv)

    print("\nSGRID compliant and Non-dimension Z-coords.\n")
    url = ("http://geoport.whoi.edu/thredds/dodsC/clay/usgs/users/"
           "jcwarner/Projects/Sandy/triple_nest/00_dir_NYB05.ncml")

    sgrid = load_ecube(url)
    verbose(sgrid)

    print("\nUGRID compliant and Non-dimension Z-coords.\n")
    url = ("http://crow.marine.usf.edu:8080/thredds/dodsC/"
           "FVCOM-Nowcast-Agg.nc")

    ugrid = load_ecube(url)
    verbose(ugrid)

    print("\nObservation.\n")
    url = "http://129.252.139.124/thredds/dodsC/fit.sispnj.met.nc"
    try:
        obs = load_ecube(url, standard_name="sea_water_temperature")
    except ValueError as e:
        print(e)
