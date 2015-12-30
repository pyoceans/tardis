from __future__ import division, absolute_import

import iris
import numpy as np
import numpy.ma as ma


iris.FUTURE.netcdf_promote = True
iris.FUTURE.cell_datetime_objects = True


__all__ = ['cf_name_list',
           'wrap_lon180',
           'wrap_lon360',
           'is_water']

salinity = ['sea_water_salinity',
            'sea_surface_salinity',
            'sea_water_absolute_salinity',
            'sea_water_practical_salinity']

temperature = ['sea_water_temperature',
               'sea_surface_temperature',
               'sea_water_potential_temperature',
               'equivalent_potential_temperature',
               'sea_water_conservative_temperature',
               'pseudo_equivalent_potential_temperature']

water_level = ['sea_surface_height',
               'sea_surface_elevation',
               'sea_surface_height_above_geoid',
               'sea_surface_height_above_sea_level',
               'water_surface_height_above_reference_datum',
               'sea_surface_height_above_reference_ellipsoid']

speed_direction = ['sea_water_speed', 'direction_of_sea_water_velocity']

u = ['surface_eastward_sea_water_velocity',
     'eastward_sea_water_velocity',
     'sea_water_x_velocity',
     'x_sea_water_velocity',
     'eastward_transformed_eulerian_mean_velocity',
     'eastward_sea_water_velocity_assuming_no_tide']

v = ['northward_sea_water_velocity',
     'surface_northward_sea_water_velocity',
     'sea_water_y_velocity',
     'y_sea_water_velocity',
     'northward_transformed_eulerian_mean_velocity',
     'northward_sea_water_velocity_assuming_no_tide']

cf_name_list = dict(
    {'salinity': salinity,
     'sea_water_temperature': temperature,
     'currents': dict(u=u, v=v, speed_direction=speed_direction),
     'water_surface_height_above_reference_datum': water_level})


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
