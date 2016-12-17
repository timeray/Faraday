"""
Consist of different ionospheric models
"""

from matplotlib import pyplot as plt
from matplotlib import rc
import numpy as np
import locale

earth_radii = 6371000  # m


def parabolic(distances, beam_angle=0, lower_height=60e3,
              max_height=300e3, maxNe=1e11):
    """
    Model electron density parabolic layer in spherically symmetric ionosphere

    @param distances: NumPy array or scalar - Distance from transmitter, m
    @param beam_angle:
            angle between normal to radar and direction of beam, deg
            positive angle is angle to equator
    @param lower_height: Starting height of the layer, m
    @param max_height: Height where electron density reach maximum, m
    @param maxNe: Max electron density at the height max_height, m^-3

    @return: NumPy array or scalar - Electron density, m^-3
    """

    # Calculate "under-ionospheric distance" using cosine theorem
    distances = -earth_radii + np.sqrt(
        earth_radii ** 2
        + distances ** 2
        + 2 * earth_radii * distances * np.cos(np.deg2rad(beam_angle))
    )

    el_density = maxNe * (1 - (((max_height - distances) ** 2)
                               / (max_height - lower_height) ** 2))

    el_density = np.array(el_density)
    el_density[el_density < 0] = 0
    if el_density.size == 1:
        return float(el_density)
    else:
        return el_density


def chapman(distances, beam_angle=0, max_height=300e3, maxNe=1e11):
    """
    Model electron density Chapman's layer in spherically symmetric atmosphere

    @param distances: NumPy array or scalar - Distance from transmitter, m
    @param beam_angle:
            angle between normal to radar and direction of beam, deg
            positive angle is angle to equator
    @param max_height: Height where electron density reach maximum, m
    @param maxNe: Max electron density at the height max_height, m^-3

    @return: NumPy array or scalar - Electron density, m^-3
    """

    # Calculate "under-ionospheric distance" using cosine theorem
    distances = -earth_radii + np.sqrt(
        earth_radii ** 2
        + distances ** 2
        + 2 * earth_radii * distances * np.cos(np.deg2rad(beam_angle))
    )

    scale_height = 30000.   # todo: calculate it from MSIS?
    y = (distances - max_height) / scale_height
    el_density = maxNe * np.exp(0.5 * (1 - y - np.exp(-y)))
    return el_density

if __name__ == '__main__':
    locale.setlocale(locale.LC_ALL, 'russian')
    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 16}
    font_axes = {'titlesize': 16}
    rc('font', **font)
    rc('axes', **font_axes)

    beg_h = 0
    end_h = 1e6

    height = np.linspace(beg_h, end_h, 1000)
    Ne1 = parabolic(height)
    Ne2 = chapman(height)

    plt.plot(Ne1, height)
    plt.plot(Ne2, height)
    plt.xlim([0, np.max([Ne1, Ne2])])

    plt.title('Профили электронной концентрации')
    plt.xlabel('Электронная концентрация, 'r'$м^{-3}$')
    plt.ylabel('Высота, км')
    plt.yticks(np.arange(beg_h, end_h + 1, 100e3),
               [int(h / 1e3) for h in np.arange(beg_h, end_h + 1, 100e3)])
    plt.legend(['Параболический слой', 'Модель Чапмена'], fontsize=16)
    plt.grid()
    plt.show()