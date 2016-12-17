"""
Collect functions to model dipole Earth magnetic field

field_intensity(lat, radar_zenith_angle, distance)
field_angle(lat, radar_zenith_angle, distance)

visualize_field()
visualize_radar(lat, radar_zenith_angle, distance)

Functions to transform geographic coordinates to geomagnetic and vice versa
geog2geom(lat, long)
geom2geog(lat, long)
"""

from matplotlib import pyplot as plt
import numpy as np

earth_radii = 6371000  # m


def field_angle(lat, radar_zenith_angle, distance):
    """
    Calculate angle between line of magnetic field and radar beam
    using dipole magnetic field approximation

    @param lat: radar geomagnetic latitude, deg, [-90:90]
    @param radar_zenith_angle:
            angle between normal to radar and direction of beam, deg
            positive angle is angle to equator
    @param distance: distance between radar and magnetic line, m
    It shouldn't (and in fact it isn't) depend on distance from radar
    if radar_zenith_angle = 0 so distance could be arbitrary.
    TODO: find formula without distance for radar_zenith_angle = 0 case

    @return: desired angle, deg
    """

    lat = np.deg2rad(lat)
    radar_zenith_angle = np.deg2rad(radar_zenith_angle)

    radar_coord = np.array([earth_radii * np.cos(lat),
                            earth_radii * np.sin(lat)])

    # If distance = 0 we move radar_coord back a little bit to have
    # non-zero radar_intersection_vector
    if distance == 0:
        radar_coord -= np.array([np.cos(lat), np.sin(lat)])

    dist_to_intersection, angle_to_intersection = calc_intersection(
        lat, radar_zenith_angle, distance
    )

    intersection_coord = np.array(
        [dist_to_intersection * np.cos(angle_to_intersection),
         dist_to_intersection * np.sin(angle_to_intersection)]
    )
    radar_intersection_vector = intersection_coord - radar_coord

    # slope of magnetic lines at latitude
    slope = ((2 * np.tan(angle_to_intersection) / 3)
             - (np.tan(np.pi / 2 - angle_to_intersection) / 3))
    slope_vector = np.array([1, slope])

    # Angle from vector dot product
    cos_of_angle = (
        np.dot(slope_vector, radar_intersection_vector)
        / np.linalg.norm(slope_vector)
        / np.linalg.norm(radar_intersection_vector)
    )

    # Due to calculation floats errors the 1 could be like 1.0000000000001
    # so we round cosine to few decimals
    cos_of_angle = np.round(cos_of_angle, 5)

    angle = np.arccos(cos_of_angle)

    # Angle with magnetic field should not be obtuse (since cos in formula for
    # Faraday rotation is negative for obtuse angles -> cos(pi - a) = -cos(a))
    # But it could be calculated as obtuse for some angles, so we do a trick:
    if angle > (np.pi / 2):
        angle = np.pi - angle

    angle = np.rad2deg(angle)
    return angle


def field_intensity(lat, radar_zenith_angle, distance):
    """
    Calculate intensity of magnetic field [A/m] at the distance
    from radar

    @param lat: radar geomagnetic latitude, deg, [-90:90]
    @param radar_zenith_angle:
            angle between normal to radar and direction of beam, deg
            positive angle is angle to equator
    @param distance: distance between radar and magnetic line, m

    @return: value of H, A/m
    """

    lat = np.deg2rad(lat)
    radar_zenith_angle = np.deg2rad(radar_zenith_angle)

    dist_to_intersection, angle_to_intersection = calc_intersection(
        lat, radar_zenith_angle, distance
    )

    intensity = (8.3e22 / 4 / np.pi
                 * np.sqrt(1 + 3 * np.sin(angle_to_intersection) ** 2)
                 / dist_to_intersection ** 3)
    return intensity


def visualize_field(axes=None):
    """
    Plot earth with dipole field (few magnetic lines). Earth radii is 1.
    @axes: Matplotlib axes with polar projection (!)

    Remember to use plt.show() to show plotted figure
    """
    if axes is None:
        axes = plt.subplot(111, projection='polar')

    theta = np.linspace(0, 2 * np.pi, 200)
    earth = [1 for x in theta]
    radii_list = [3, 5, 6, 10]
    field_lines = [[rad * np.cos(x) ** 2 for x in theta] for rad in radii_list]
    plt.plot(theta, earth, color='red', lw=1.5)
    for field_line in field_lines:
        plt.plot(theta, field_line, color='blue', lw=1.25)

    axes.set_yticklabels('', visible=False)  # remove range labels
    theta_ticks = np.arange(0, 360, 45)
    theta_tick_labels = [0, 45, 90, 45, 0, -45, -90, -45]
    # Degrees
    # theta_tick_labels = ['%d'r'$\mathrm{^{o}}$'
    theta_tick_labels = ['%d'r'$\degree$'
                         % label for label in theta_tick_labels]
    axes.set_thetagrids(theta_ticks, theta_tick_labels, frac=1.1)


def visualize_radar(lat, radar_zenith_angle, distance, axes=None):
    """
    Plot dipole field and radar at lat latitude [deg] with beam pointed
    radar_zenith_angle angle [deg] away from normal. Earth radii is 1.

    distance - distance from radar, m
    @axes - matplotlib axes with polar projection (!)

    Remember to use plt.show() to show plotted figure
    """
    if axes is None:
        axes = plt.subplot(111, projection='polar')

    theta = np.linspace(0, 2 * np.pi, 200)

    earth = [1 for x in theta]
    axes.plot(theta, earth, color='red', lw=1.5)
    # Limit possible distance
    axes.set_ylim(0, 3)

    lat = np.deg2rad(lat)
    radar_zenith_angle = np.deg2rad(radar_zenith_angle)

    dist_to_intersection, angle_to_intersection = calc_intersection(
        lat, radar_zenith_angle, distance
    )

    # If angle is near 0 deg
    epsilon = 0.01
    if ((angle_to_intersection >= (np.pi / 2 - epsilon))
            & (angle_to_intersection <= (np.pi / 2 + epsilon))):
        print('Line is undrawable for low angles')

    norm_dist = dist_to_intersection / earth_radii

    # Radar lines
    axes.plot([lat, lat], [0, 1], color='green', lw=1.5)
    axes.plot([lat, angle_to_intersection], [1, norm_dist],
              color='green', lw=1.5)

    # Magnetic line
    equator_dist = norm_dist / np.cos(angle_to_intersection) ** 2
    axes.plot(theta, [equator_dist * np.cos(x) ** 2 for x in theta],
              color='blue', lw=1.5)


def calc_intersection(lat, radar_zenith_angle, distance):
    """
    Calculate polar coordinated of intersection between radar beam and
    magnetic line of dipole field

    @param lat: latitude of radar, rad (!) not deg here)
    @param radar_zenith_angle:
            angle between normal to radar and direction of beam, deg
            positive angle is angle to equator
    @param distance: distance from radar, m

    @return: tuple - polar distance angle to intersection
    """

    if (distance != 0) & (radar_zenith_angle != 0):
        dist_to_intersection = np.sqrt(earth_radii ** 2 + distance ** 2
                                       + 2 * earth_radii * distance
                                       * np.cos(radar_zenith_angle))
        # radar_zenith_angle lies in range [-90:90] and for negatives
        # angle_radar_intersection should also be negative
        angle_radar_intersection = np.sign(radar_zenith_angle) * np.arccos(
            (earth_radii ** 2 + dist_to_intersection ** 2 - distance ** 2)
            / (2 * earth_radii * dist_to_intersection)
        )
        angle_to_intersection = lat - angle_radar_intersection
    else:
        dist_to_intersection = earth_radii + distance
        angle_to_intersection = lat

    return dist_to_intersection, angle_to_intersection


def geog2geom(lat, long):
    """
    Rough transform geographical latitude and longitude to geomagnetic
    for position of the pole at 2015 (!)

    Latitude range is [-90:90] where positive is North, negative - South
    Longitude range is [-180:180] where positive is East, negative - West

    @param lat: Latitude, deg
    @param long: Longitude, deg
    @return: tuple: (latitude, longitude), deg
    """
    lat = np.deg2rad(lat)
    long = np.deg2rad(long)

    # Pole coordinates for 2015
    pole_lat = np.deg2rad(80.37)
    pole_long = np.deg2rad(-72.62)

    pole_lat_s = np.sin(pole_lat)
    pole_lat_c = np.cos(pole_lat)
    pole_long_s = np.sin(pole_long)
    pole_long_c = np.cos(pole_long)

    # Rotation matrix
    matrix = np.array([
        [pole_lat_s * pole_long_c, pole_lat_s * pole_long_s, -pole_lat_c],
        [-pole_long_s, pole_long_c, 0],
        [pole_lat_c * pole_long_c, pole_lat_c * pole_long_s, pole_lat_s]
    ])

    x = earth_radii * np.cos(lat) * np.cos(long)
    y = earth_radii * np.cos(lat) * np.sin(long)
    z = earth_radii * np.sin(lat)
    vect_geog = np.array([x, y, z])
    vect_geom = np.dot(matrix, vect_geog)
    norm = np.linalg.norm(vect_geom)

    lat_geom = np.arcsin(vect_geom[2] / norm)
    long_geom = np.arctan2(vect_geom[1], vect_geom[0])

    lat_geom = np.rad2deg(lat_geom)
    long_geom = np.rad2deg(long_geom)
    return lat_geom, long_geom


def geom2geog(lat, long):
    """
    Rough transform geomagnetic latitude and longitude to geographic
    for position of the pole at 2015 (!)

    Latitude range is [-90:90] where positive is North, negative - South
    Longitude range is [-180:180] where positive is East, negative - West

    @param lat: Latitude, deg
    @param long: Longitude, deg
    @return: tuple: (latitude, longitude), deg
    """
    lat = np.deg2rad(lat)
    long = np.deg2rad(long)

    # Pole coordinates for 2015
    pole_lat = np.deg2rad(80.37)
    pole_long = np.deg2rad(-72.62)

    pole_lat_s = np.sin(pole_lat)
    pole_lat_c = np.cos(pole_lat)
    pole_long_s = np.sin(pole_long)
    pole_long_c = np.cos(pole_long)

    # Rotation matrix
    matrix = np.array([
        [pole_lat_s * pole_long_c, pole_lat_s * pole_long_s, -pole_lat_c],
        [-pole_long_s, pole_long_c, 0],
        [pole_lat_c * pole_long_c, pole_lat_c * pole_long_s, pole_lat_s]
    ])
    matrix = np.linalg.inv(matrix)

    x = earth_radii * np.cos(lat) * np.cos(long)
    y = earth_radii * np.cos(lat) * np.sin(long)
    z = earth_radii * np.sin(lat)
    vect_geom = np.array([x, y, z])
    vect_geog = np.dot(matrix, vect_geom)
    norm = np.linalg.norm(vect_geog)

    lat_geog = np.arcsin(vect_geog[2] / norm)
    long_geog = np.arctan2(vect_geog[1], vect_geog[0])

    lat_geog = np.rad2deg(lat_geog)
    long_geog = np.rad2deg(long_geog)
    return lat_geog, long_geog


if __name__ == '__main__':
    test_latitudes = [0, 10, 45, 75, -10, -25, 90]
    test_inclination = [45, 0, -20, 10, -30, 25, -45]
    test_distance = 1e6
    for lat_mag, inclination in zip(test_latitudes, test_inclination):
        visualize_radar(lat_mag, inclination, test_distance)
        print('For radar at %.3f deg with beam inclination %d,'
              ' angle is %.3f while intensity is %.3f'
              % (lat_mag, inclination,
                 field_angle(lat_mag, inclination, test_distance),
                 field_intensity(lat_mag, inclination, test_distance)))

    test_geo = [[0, 0], [20, -20], [-50, 130], [70, 10]]
    for test in test_geo:
        lat_mag, long_mag = geog2geom(test[0], test[1])
        lat_geo, long_geo = geom2geog(lat_mag, long_mag)
        print('\nTest GEO lat %.3f long %.3f'
              '\nGeomagnetic: lat %.3f, long %.3f'
              '\nGeographic:  lat %.3f, long %.3f'
              % (test[0], test[1], lat_mag, long_mag, lat_geo, long_geo))

    plt.show()
