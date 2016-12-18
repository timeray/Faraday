"""
Main file for plotting major dependencies
"""

from scipy import constants
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import locale
import os
import magfield as mf
import ionosphere as io
import faraday

if __name__ == '__main__':
    # File tunes
    savedir = os.path.join(os.path.dirname(__file__), 'Figures')

    # Fonts and language
    locale.setlocale(locale.LC_ALL, 'russian')
    font_size = 16
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': font_size}
    font_axes = {'titlesize': font_size}
    rc('font', **font)
    rc('axes', **font_axes)

    # Calculations
    num_of_iterations = 200

    # Model general parameters
    altitude_boundary = 1000e3
    radar_frequency = 158e6
    lat_geog = 53
    long_geog = 106
    lat_geom, long_geom = mf.geog2geom(lat_geog, long_geog)

    beam_angle = 0      # Angle between beam and normal to radar plane
    pulse_len = 50e-6   # Pulse length, seconds
    max_power = 1       # Pulse peak power, W
    gain = 1            # Radar antenna gain
    max_density = 5e11  # Maximal electron density for ionospheric models, m^-3
    ratio_of_temps = 1  # Ratio of election and ion temperatures

    ionosphere_func = io.parabolic  # Function for ionosphere model

    # Plotting
    fig_size = (6, 4.8)
    fig_size_large = (8, 6)
    figures = {}

    # Plotting functions. See Calls below
    # --- Ionosphere ---
    def plot_ionosphere(el_density_max=max_density, axes=None):
        print('Plotting ionosphere...')
        if axes is None:
            figures['ionosphere'] = plt.figure('ionosphere', figsize=fig_size)
            axes = plt.subplot(111)

        heights = np.linspace(0, altitude_boundary, num_of_iterations)
        el_density = np.zeros(heights.shape)
        for i, alt in enumerate(heights):
            el_density[i] = ionosphere_func(alt, beam_angle,
                                            maxNe=el_density_max)

        axes.plot(el_density, heights)
        axes.set_xlabel('Электронная концентрация, м'r'$^{-3}$')
        axes.set_ylabel('Высота, км')
        yticks = np.arange(0, altitude_boundary + 1, 100e3)
        plt.yticks(yticks, [int(tick / 1e3) for tick in yticks])
        axes.xaxis.set_major_formatter(FormatStrFormatter('%1.e'))
        # plt.title('Модель электронной концентрации в ионосфере')
        axes.grid()
        plt.tight_layout()
        return axes


    # --- General magnetic field ---
    def plot_dipole_field():
        print('Plotting dipole_field...')
        figures['dipole_field'] = plt.figure('dipole_field', figsize=fig_size)
        ax = figures['dipole_field'].add_subplot(111, projection='polar')
        mf.visualize_field(ax)

    # --- Magnetic field with beams ---
    def plot_dipole_beams():
        print('Plotting dipole_beams...')
        figures['dipole_beams'] = plt.figure('dipole_beams', figsize=fig_size)
        ax = figures['dipole_beams'].add_subplot(111, projection='polar')
        latitudes = [0, 10, 45, 75, -10, -25, 90]
        inclinations = [45, 0, -45, -10, -30, 25, 45]
        distances = [2000e3, 5000e3, 1200e3, 2000e3, 200e3, 500e3, 1000e3]

        for lat, incl, dist in zip(latitudes, inclinations, distances):
            mf.visualize_radar(lat, incl, dist, ax)

        ax.set_yticklabels('', visible=False)  # remove range labels
        theta_ticks = np.arange(0, 360, 45)
        theta_tick_labels = [0, 45, 90, 45, 0, -45, -90, -45]
        # Degrees
        # theta_tick_labels = ['%d'r'$\mathrm{^{o}}$'
        theta_tick_labels = ['%d'r'$\degree$'
                             % label for label in theta_tick_labels]
        ax.set_thetagrids(theta_ticks, theta_tick_labels, frac=1.15)
        plt.tight_layout()

    # --- Pure Faraday rotation angle dependence on angle to magnetic field ---
    # and with change in frequency
    def angle_dependence():
        print('Plotting angle_dependence...')
        figures['angle_dependence'] = plt.figure('angle_dependence',
                                                 figsize=fig_size)
        ax = plt.subplot(111)

        angles = np.linspace(0, 90, num_of_iterations)
        frequencies = [100e6, 150e6, 300e6]
        rotation_angles = np.zeros(angles.shape)
        for freq in frequencies:
            for i, angle_to_field in enumerate(angles):
                # simple field_angle function, just returning angle_to_field
                # angle_fo_field in degrees, rotation_angles in radians
                rotation_angles[i] = faraday.faraday_rotation(
                    altitude_boundary, freq,
                    lambda dist: ionosphere_func(dist, beam_angle,
                                                 maxNe=max_density),
                    lambda dist: mf.field_intensity(lat_geom, beam_angle, dist),
                    lambda dist: angle_to_field,
                    optimize=True
                )
            rotations = rotation_angles / (2 * np.pi)
            ax.plot(angles, rotations, lw=1.5)

        # plt.title('Зависимость Фарадеевского вращения от угла к магнитному полю'
        #           '\nрасстояние от поверхности - %d км, максимум Ne - %1.e м'
        #           r'$^{-3}$'
        #           % (int(altitude_boundary / 1e3), max_density))
        plt.legend(['%d МГц' % (freq / 1e6) for freq in frequencies],
                   title='Частота', fontsize=font_size)
        ax.set_xlabel('Угол к магнитному полю, град')
        ax.set_ylabel('Число полных оборотов поляризации 'r'$(2\pi)$')
        ax.grid()
        plt.tight_layout()


    # --- Pure Faraday rotation dependence on geomagnetic latitude ---
    def latitude_dependence():
        print('Plotting latitude_dependence...')
        figures['latitude_dependence'] = plt.figure('latitude_dependence',
                                                    figsize=fig_size_large)
        ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)

        latitudes = np.linspace(0, 90, num_of_iterations)
        frequencies = [100e6, 150e6, 300e6]
        rotation_angles = np.zeros(latitudes.shape)
        for freq in frequencies:
            for i, lat in enumerate(latitudes):
                rotation_angles[i] = faraday.faraday_rotation(
                    altitude_boundary, freq,
                    lambda dist: ionosphere_func(dist, beam_angle,
                                                 maxNe=max_density),
                    lambda dist: mf.field_intensity(lat, beam_angle, dist),
                    lambda dist: mf.field_angle(lat, beam_angle, dist),
                    optimize=True
                )
            rotations = rotation_angles / (2 * np.pi)
            ax1.plot(latitudes, rotations, lw=1.5)

        # # Title for all subplots
        # figures['latitude_dependence'].suptitle(
        #     'Зависимость Фарадеевского вращения от широты\nрасстояние'
        #     ' от поверхности - %d км, максимум Ne - %1.e м'r'$^{-3}$'
        #     % (int(altitude_boundary / 1e3), max_density)
        # )
        # plt.subplots_adjust(top=0.85)  # space for title
        plt.legend(['%d МГц' % (freq / 1e6) for freq in frequencies],
                   title='Частота', fontsize=font_size, framealpha=0.75)
        ax1.set_xlabel('Геомагнитная широта, град')
        ax1.invert_xaxis()
        ax1.set_xticks([tick for tick in range(0, 91, 15)])
        ax1.set_ylabel('Число полных оборотов поляризации 'r'$(2\pi)$')
        ax1.grid()

        # Magnetic field showing angles
        ax2 = plt.subplot2grid((2, 2), (0, 1), projection='polar')
        num_of_lines = 6
        slicer = int(num_of_iterations / num_of_lines)
        for lat in latitudes[::slicer]:
            mf.visualize_radar(lat, beam_angle, altitude_boundary, ax2)

        ax2.set_yticklabels('', visible=False)  # remove range labels
        theta_ticks = np.arange(0, 360, 45)
        theta_tick_labels = [0, 45, 90, 45, 0, -45, -90, -45]
        # Degrees
        # theta_tick_labels = ['%d'r'$\mathrm{^{o}}$'
        theta_tick_labels = ['%d'r'$\degree$'
                             % label for label in theta_tick_labels]
        ax2.set_thetagrids(theta_ticks, theta_tick_labels, frac=1.15)

        # Dependence of angle to magnetic field from latitude
        ax3 = plt.subplot2grid((2, 2), (1, 1))
        angles_to_magfield = np.zeros(latitudes.shape)
        for i, lat in enumerate(latitudes):
            angles_to_magfield[i] = mf.field_angle(lat, beam_angle,
                                                   altitude_boundary)

        ax3.plot(latitudes, angles_to_magfield, lw=1.5)
        ax3.plot(latitudes, latitudes[::-1], 'b--')     # y = x line
        ax3.invert_xaxis()
        ax3.set_xlabel('Геомагнитная широта, град')
        ax3.set_ylabel('Угол к магнитному полю')
        ax3.set_xticks([tick for tick in range(0, 91, 15)])
        ax3.set_yticks([tick for tick in range(0, 91, 15)])
        ax3.yaxis.tick_right()
        ax3.yaxis.set_label_position('right')
        ax3.grid()
        plt.tight_layout()

    # --- Beaming with fixed angle to field ---
    def beaming():
        print('Plotting beaming...')
        figures['beaming'] = plt.figure('beaming', figsize=fig_size)
        ax = figures['beaming'].add_subplot(111)

        angle_sector = np.linspace(-90, 90, num_of_iterations)
        # And for different distances from transmitter
        distances = [250e3, 500e3, 1000e3, 20e6]
        angle_to_field = 0

        rotation_angles = np.zeros(angle_sector.shape)
        for distance in distances:
            for i, angle in enumerate(angle_sector):
                # We interested only in ionospheric angle dependence
                rotation_angles[i] = faraday.faraday_rotation(
                    distance, radar_frequency,
                    lambda dist: ionosphere_func(dist, angle,
                                                 maxNe=max_density),
                    lambda dist: mf.field_intensity(lat_geom, angle_to_field,
                                                    dist),
                    lambda dist: mf.field_angle(lat_geom, angle_to_field,
                                                dist),
                    optimize=True
                )
            rotations = rotation_angles / 2 * np.pi  # rotations
            ax.plot(angle_sector, rotations, lw=1.5)

        ax.set_xlabel('Угол наклона луча')
        ax.set_ylabel('Число оборотов вектора поляризации 'r'$(2\pi)$')
        ax.grid()
        plt.xlim([-90, 90])
        plt.xticks(np.arange(-90, 91, 45))
        plt.legend(['%d км' % int(dist / 1e3) for dist in distances],
                   framealpha=65)
        plt.tight_layout()

    # --- Faraday rotation on beam angle and latitude of transmitter ---
    def beam_latitude():
        print('Plotting beam_latitude...')
        figures['beam_latitude'] = plt.figure('beam_latitude', figsize=fig_size)
        ax = figures['beam_latitude'].add_subplot(111)

        # Boundaries of angle sector
        low_lim_angle = -75
        upp_lim_angle = 75

        # Upper height
        altitude_boundary = 10000e3

        num_of_points = int(num_of_iterations / 5)   # time saving
        latitudes = np.linspace(0, 90, num_of_points)
        angle_sector = np.linspace(low_lim_angle, upp_lim_angle, num_of_points)
        rotation_angles = np.zeros((latitudes.size, angle_sector.size))
        for i, lat in enumerate(latitudes):
            print('For latitude ', lat)
            for j, angle in enumerate(angle_sector):
                rotation_angles[i, j] = faraday.faraday_rotation(
                    altitude_boundary, radar_frequency,
                    lambda dist: ionosphere_func(dist, angle,
                                                 maxNe=max_density),
                    lambda dist: mf.field_intensity(lat, angle, dist),
                    lambda dist: mf.field_angle(lat, angle, dist),
                    optimize=True
                )

        levels = 20
        contourf = ax.contourf(latitudes, angle_sector,
                               rotation_angles.T, levels)
        ax.contour(latitudes, angle_sector, rotation_angles.T,
                   levels, colors='black', lw=0.7)
        ax.set_xlabel('Геомагнитная широта')
        ax.set_ylabel('Угол наклона луча')
        ax.set_yticks(np.arange(low_lim_angle, upp_lim_angle + 1, 15))
        colorbar = plt.colorbar(contourf)
        colorbar.set_label('Угол поворота плоскости поляризации, рад')
        plt.tight_layout()


    # --- Radar received power ---
    def received_power():
        print('Plotting received_power...')
        figures['received_power'] = plt.figure('received_power',
                                               figsize=fig_size)
        axes = []

        max_density = 1e12

        # We need to call faraday.received_power which takes, so we need
        # closures of functions to make them dependent on altitude only
        def closure_omega():
            def omega(omega_alt):
                return faraday.faraday_rotation(
                    omega_alt, radar_frequency,
                    lambda dist: io.parabolic(dist, maxNe=max_density),
                    lambda dist: mf.field_intensity(lat_geom, beam_angle, dist),
                    lambda dist: mf.field_angle(lat_geom, beam_angle, dist),
                    optimize=True
                )
            return omega

        def closure_power_delta():
            def power_delta(power_alt):
                # Setting radar params here
                return faraday.received_power_delta(
                    power_alt, max_power, gain, radar_frequency,
                    lambda dist: ratio_of_temps,
                    lambda dist: io.parabolic(dist, maxNe=max_density),
                    closure_omega()
                )
            return power_delta

        # Now we can call faraday.received_power
        pulse_durations = [50e-6, 250e-6, 700e-6]

        time_boundary = 2 * altitude_boundary / constants.speed_of_light
        receiver_time = np.linspace(0, time_boundary, num_of_iterations)
        overall_power = np.zeros(receiver_time.shape)
        time_ticks = np.arange(0, time_boundary + 0.001, 0.001)

        # Create xlabels on top for equivalent distance
        ax_dist = 0
        distances = np.arange(0, altitude_boundary + 100000, 100000)
        distance_labels = ['%d' % int(dist / 1000) for dist in distances]
        distance_ticks = 2 * distances / constants.speed_of_light

        for i, pulse_duration in enumerate(pulse_durations):
            ax = plt.subplot(3, 1, i + 1)
            axes.append(ax)
            for j, tau in enumerate(receiver_time):
                print('Iteration: %d/%d' % ((j + 1), num_of_iterations))
                overall_power[j] = faraday.received_power(
                    tau,
                    pulse_duration,
                    closure_power_delta(),
                    lambda dist: faraday.rect_envelope(dist, pulse_duration)
                )

            ax.plot(receiver_time, overall_power)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_ylabel(r'$P_п$')

            # Set legend and remove blue line from it
            leg = plt.legend(
                [r'$\tau_{и}$'' = %d 'r'$\mu s$' % int(pulse_duration * 1e6)],
                handlelength=0, handletextpad=0
            )
            for item in leg.legendHandles:
                item.set_visible(False)

            # Distance xaxis at first plot
            if ax == axes[0]:
                ax_dist = ax.twiny()
                ax_dist.plot(receiver_time, overall_power)

        axes[-1].set_xticks(time_ticks)
        axes[-1].set_xticklabels(['%d' % int(tick * 1000)
                                  for tick in time_ticks])
        axes[-1].set_xlabel('Время, мс')

        ax_dist.set_xticks(distance_ticks)
        ax_dist.set_xticklabels(distance_labels)
        ax_dist.set_xlabel('Дальность от передатчика, км')

        plt.tight_layout()

    # --- Calls. Uncomment functions you want to plot ---
    # plot_ionosphere()
    # angle_dependence()
    # latitude_dependence()
    # plot_dipole_field()
    # plot_dipole_beams()
    # beaming()
    # beam_latitude()
    received_power()

    # After all plots
    for key in figures:
        figures[key].savefig(os.path.join(savedir, key))

    plt.show()
