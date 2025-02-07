import matplotlib.pyplot as plt
import numpy as np

def main():
    particles_per_cell = np.array([16, 32, 64])

    absolute_error1 = np.array([
        7.629394602304274e-06,
        1.7166137702417927e-05,
        4.5776367201710855e-05
    ])
    absolute_error2 = np.array([
        7.62939459875156e-06,
        1.9073486399179274e-05,
        5.340576180401513e-05
    ])
    absolute_error3 = np.array([
        7.629394602304274e-06,
        2.2888183600855427e-05,
        6.1035156207367436e-05
    ])
    absolute_error4 = np.array([
        1.1444091800427714e-05,
        2.6702880795426154e-05,
        7.629394529828915e-05
    ])


    plt.plot(particles_per_cell, absolute_error1, 'o-')
    plt.plot(particles_per_cell, absolute_error2, 'o-')
    plt.plot(particles_per_cell, absolute_error3, 'o-')
    plt.plot(particles_per_cell, absolute_error4, 'o-')

    plt.xticks(particles_per_cell)
    plt.xlabel('Particles per cell')
    plt.ylabel('Max absolute error')
    plt.legend(['32x32', '64x64', '128x128', '256x256'], title='Grid cells')
    plt.title('Charge density error')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
