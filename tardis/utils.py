from __future__ import division, absolute_import

import numpy as np
import numpy.ma as ma

import iris
from iris.exceptions import CoordinateNotFoundError


iris.FUTURE.netcdf_promote = True
iris.FUTURE.cell_datetime_objects = True


__all__ = ['wrap_lon180',
           'wrap_lon360',
           'is_model',
           'is_water']


def wrap_lon180(lon):
    lon = np.atleast_1d(lon).copy()
    angles = np.logical_or((lon < -180), (180 < lon))
    lon[angles] = wrap_lon360(lon[angles] + 180) - 180
    return lon


def wrap_lon360(lon):
    lon = np.atleast_1d(lon).copy()
    positive = lon > 0
    lon = lon % 360
    lon[np.logical_and(lon == 0, positive)] = 360
    return lon


def _source_of_data(cube, coverage_content_type='modelResult'):
    """
    Check if the `coverage_content_type` of the cude.
    The `coverage_content_type` is an ISO 19115-1 code to indicating the
    source of the data types and can be one of the following:

    image, thematicClassification, physicalMeasurement, auxiliaryInformation,
    qualityInformation, referenceInformation, modelResult, coordinate

    Examples
    --------
    >>> import iris
    >>> iris.FUTURE.netcdf_promote = True
    >>> url = ("http://comt.sura.org/thredds/dodsC/data/comt_1_archive/"
    ...        "inundation_tropical/VIMS_SELFE/"
    ...        "Hurricane_Ike_2D_final_run_without_waves")
    >>> cubes = iris.load_raw(url, 'sea_surface_height_above_geoid')
    >>> [_source_of_data(cube) for cube in cubes]
    [False, True]
    """

    attributes = cube.attributes
    cube_coverage_content_type = attributes.get('coverage_content_type', None)
    if cube_coverage_content_type == coverage_content_type:
        return True
    return False


def is_model(cube):
    """
    Heuristic way to find if a cube data is `modelResult` or not.
    WARNING: This function may return False positives and False
    negatives!!!

    Examples
    --------
    >>> import iris
    >>> iris.FUTURE.netcdf_promote = True
    >>> url = ("http://crow.marine.usf.edu:8080/thredds/dodsC/"
    ...        "FVCOM-Nowcast-Agg.nc")
    >>> cubes = iris.load_raw(url, 'sea_surface_height_above_geoid')
    >>> [is_model(cube) for cube in cubes]
    [True]
    >>> url = ("http://thredds.cdip.ucsd.edu/thredds/dodsC/cdip/archive/"
    ...        "043p1/043p1_d17.nc")
    >>> cubes = iris.load_raw(url, 'sea_surface_temperature')
    >>> [is_model(cube) for cube in cubes]
    [False]

    """
    # First criteria (Strong): "forecast" word in the time coord.
    try:
        coords = cube.coords(axis='T')
        for coord in coords:
            if 'forecast' in coord.name():
                return True
    except CoordinateNotFoundError:
        pass
    # Second criteria (Strong): `UGRID` cubes are models.
    conventions = cube.attributes.get('Conventions', 'None')
    if 'UGRID' in conventions.upper():
        return True
    # Third criteria (Strong): dimensionless coords are present.
    try:
        coords = cube.coords(axis='Z')
        for coord in coords:
            if 'ocean_' in coord.name():
                return True
    except CoordinateNotFoundError:
        pass
    # Forth criteria (weak): Assumes that all "GRID" attribute are models.
    cdm_data_type = cube.attributes.get('cdm_data_type', 'None')
    feature_type = cube.attributes.get('featureType', 'None')
    source = cube.attributes.get('source', 'None')
    if cdm_data_type.upper() == 'GRID' or feature_type.upper() == 'GRID':
        if 'AVHRR' not in source:
            return True
    return False


def is_water(cube, min_var=0.01):
    """
    Use only data where the standard deviation of the time cube exceeds
    0.01 m (1 cm) this eliminates flat line model time cube that come from
    land points that should have had missing values.
    (Accounts for wet-and-dry models.)

    """
    arr = ma.masked_invalid(cube.data).filled(fill_value=0)
    if arr.std() <= min_var:
        return False
    return True

if __name__ == '__main__':
    import doctest
    doctest.testmod()
