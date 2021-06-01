import iris
from iris.exceptions import CoordinateNotFoundError

iris.FUTURE.netcdf_promote = True
iris.FUTURE.cell_datetime_objects = True


__all__ = ["t_coord", "z_coord"]


def t_coord(cube):
    """
    Return the canonical time coordinate and rename it to `time`.
    If more than one time coordinate is present it will try to return only
    the one names `time`.

    Examples
    --------
    >>> import iris
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/"
    ...        "sabgom/SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> str(t_coord(cube).name())
    'time'

    """
    try:
        cube.coord(axis="T").rename("time")
    except CoordinateNotFoundError:
        # This will fail if there more than 1 time coordinate
        # and none are named time.
        coord = cube.coord("time")
    else:
        coord = cube.coord("time")
    return coord


def z_coord(cube):
    """
    Return the canonical vertical coordinate.

    Examples
    --------
    >>> import iris
    >>> import warnings
    >>> url = ("http://omgsrv1.meas.ncsu.edu:8080/thredds/dodsC/fmrc/"
    ...        "sabgom/SABGOM_Forecast_Model_Run_Collection_best.ncd")
    >>> cube = iris.load_cube(url, 'sea_water_potential_temperature')
    >>> str(z_coord(cube).name())
    'ocean_s_coordinate_g1'

    """
    non_dimensional = [
        "atmosphere_hybrid_height_coordinate",
        "atmosphere_hybrid_sigma_pressure_coordinate",
        "atmosphere_sigma_coordinate",
        "atmosphere_sleve_coordinate",
        "ocean_s_coordinate",
        "ocean_s_coordinate_g1",
        "ocean_s_coordinate_g2",
        "ocean_sigma_coordinate",
        "ocean_sigma_z_coordinate",
    ]
    coord = None
    # If only one exists get that.
    try:
        coord = cube.coord(axis="Z")
    except CoordinateNotFoundError:
        # If a named `z_coord` exist.
        try:
            coord = cube.coord(axis="altitude")
        except CoordinateNotFoundError:
            # OK, let's use the non-dimensional names
            # until http://cf-trac.llnl.gov/trac/ticket/143 is in.
            for coord in cube.coords(axis="Z"):
                if coord.name() in non_dimensional:
                    coord = coord
                    break
    return coord


if __name__ == "__main__":
    import doctest

    doctest.testmod()
