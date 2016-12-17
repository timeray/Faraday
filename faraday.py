"""

"""

from scipy import constants
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import rc
import locale
import numpy as np
import magfield as mf
import ionosphere as io


def faraday_rotation(upper_height, freq, el_density, magfield_intensity,
                     magfield_angle, optimize=False):
    """
    Calculate rotation of linearly polarized wave

    @param upper_height: Upper height of calculation, m
    @param freq: Wave frequency, Hz
    @param el_density: function:
            Electron density as function of height, m
    @param magfield_intensity: function:
            Intensity of magnetic field as function of distance, A/m
    @param magfield_angle: function:
            Angle to magnetic field as function of distance, deg
    @param optimize: bool:
            Optimize calculation procedure to integrate for pieces of distance,
            not a full distance each time.
            Optimization zeros if:
                1) non-optimized function called
                2) previous upper_height exceeds current upper_height
                3) previous upper_height is equal to current upper_height
            Never use it in precise calculations.

    @return: NumPy array - Profile of rotation angle along upper_height
    """
    try:
        if not optimize:
            faraday_rotation.lower_limit = 0
        else:
            if faraday_rotation.lower_limit >= upper_height:
                # Zero out counter
                # Even if limits are the same
                faraday_rotation.lower_limit = 0
                faraday_rotation.prev_integral = 0
    except AttributeError:
        faraday_rotation.lower_limit = 0
        faraday_rotation.prev_integral = 0

    lower_height = faraday_rotation.lower_limit

    def integrand(dist):
        angle = np.deg2rad(magfield_angle(dist))
        return el_density(dist) * magfield_intensity(dist) * np.cos(angle)

    integral, err = integrate.quad(integrand, lower_height,
                                   upper_height, limit=100)

    # Increase lower_limit for future
    if (optimize is True) & (faraday_rotation.lower_limit < upper_height):
        faraday_rotation.lower_limit = upper_height
        integral += faraday_rotation.prev_integral
        faraday_rotation.prev_integral = integral

    # # Integration info
    # if integral != 0:
    #     print('Rotation integral with absolute error %e'
    #           ' and relative error %e' % (err, err / integral))
    #
    # print('Value of integral is ', integral)

    cycle_freq = 2 * constants.pi * freq
    rotation_angle = (integral
                      * constants.elementary_charge ** 3
                      * constants.mu_0
                      / (2 * constants.epsilon_0 * constants.electron_mass ** 2
                         * cycle_freq ** 2 * constants.speed_of_light))
    return rotation_angle


def received_power_delta(distance, pulse_power, max_gain, freq, temp_ratio,
                         el_density, faraday_rotation):
    """
    Calculate power at the receiver from incoherent scatter of linearly
    polarized wave from delta-pulse
    @param distance: Distance from radar, m
    @param pulse_power: Pulse average power, W
    @param max_gain: Gain of radar antenna at maximum direction
    @param freq: Frequency of radar, Hz
    @param temp_ratio: function:
            Ratio of electron to ion temperature as function of range
    @param el_density: function:
            Election density as function of range, m^-3
    @param faraday_rotation: function:
            Faraday rotation angle as function of range, rad

    @return: Power scattered from delta-pulse
    """
    wave_number = 2 * np.pi * freq / constants.speed_of_light
    amplitude = pulse_power * max_gain / wave_number
    rad_e = constants.physical_constants['classical electron radius'][0]
    power_delta = (np.pi * rad_e ** 2 * amplitude * el_density(distance)
                   * np.cos(faraday_rotation(distance)) ** 2
                   / (distance ** 2 * (1 + temp_ratio(distance))))
    return power_delta


def received_power(time, pulse_duration, power_delta, envelope):
    """
    Convolution of power from incoherent scatter with pulse of pulse duration
    pulse_duration at specific time.

    @param time: Time at receiver, seconds
    @param pulse_duration: Duration of pulse, seconds
                           we need it here just to limit integration
    @param power_delta: function:
            Power of scattered delta-pulse as function of range
    @param envelope: function:
            Envelope as function of time

    @return:
    """
    c = constants.speed_of_light

    def integrand(dist):
        return power_delta(dist) * envelope(time - 2 * dist / c)

    # To decrease the amount of calculation
    # integration boundaries could be limited by pulse duration
    upper_limit = time * c / 2
    lower_limit = (time - pulse_duration) * c / 2
    power, err = integrate.quad(integrand, lower_limit, upper_limit)
    # if power != 0:
    #     print('Power integral with absolute error %e and relative error %e' %
    #           (err, err / power))
    print('Received power: ', power)
    return power


def rect_envelope(time, duration):
    """
    Pulse envelope

    @param time: Desired time, seconds
    @param duration: Duration of envelope, seconds
    @return: Envelope
    """
    if (time > 0) & (time < duration):
        return 1
    else:
        return 0


if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'russian')
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16}
    font_axes = {'titlesize': 16}
    rc('font', **font)
    rc('axes', **font_axes)

    # Model params
    altitude_boundary = 1000e3
    frequency = 158e6
    lat_geog = 53
    long_geog = 106
    lat_geom, long_geom = mf.geog2geom(lat_geog, long_geog)

    beam_angle = 0
    pulse_len = 50e-6
    max_power = 1
    gain = 1
    max_density = 1e12
    ratio_of_temps = 1

    # Closures of functions to make them depend on altitude only
    def closure_omega():
        def omega(omega_alt):
            return faraday_rotation(
                omega_alt, frequency,
                lambda dist: io.parabolic(dist, maxNe=max_density),
                lambda dist: mf.field_intensity(lat_geom, beam_angle, dist),
                lambda dist: mf.field_angle(lat_geom, beam_angle, dist),
                optimize=True
            )

        return omega


    def closure_power_delta():
        def power_delta(power_alt):
            return received_power_delta(
                power_alt, max_power, gain, frequency,
                lambda dist: ratio_of_temps,
                lambda dist: io.parabolic(dist, maxNe=max_density),
                closure_omega()
            )

        return power_delta


    # Testing
    iterations = 300
    # Faraday rotation
    altitudes = np.linspace(0, altitude_boundary, iterations)
    faraday_func = closure_omega()
    density_profile = np.zeros(altitudes.shape)
    overall_rotation = np.zeros(altitudes.shape)
    for i, alt in enumerate(altitudes):
        overall_rotation[i] = faraday_func(alt)
        density_profile[i] = io.parabolic(alt, maxNe=max_density)

    ax1 = plt.subplot(111)
    ax1.plot(overall_rotation, altitudes, color='red')
    ax1.set_ylabel('Высота, км')
    ax2 = ax1.twiny()
    ax2.plot(density_profile, altitudes, color='blue')

    # Received power
    time_boundary = 2 * altitude_boundary / constants.speed_of_light
    receiver_time = np.linspace(0, time_boundary, iterations)
    overall_power = np.zeros(receiver_time.shape)
    for i, tau in enumerate(receiver_time):
        print('Iteration: %d/%d' % ((i + 1), iterations))
        overall_power[i] = received_power(
            tau, pulse_len, closure_power_delta(),
            lambda dist: rect_envelope(dist, pulse_len)
        )

    plt.figure()
    plt.plot(receiver_time, overall_power)

    plt.show()
