import matplotlib.pyplot as plt
import numpy as np

def main():
    particle_count1 = np.array([32*32*16, 32*32*32, 32*32*64])
    particle_count2 = np.array([64*64*16, 64*64*32, 64*64*64])
    particle_count3 = np.array([128*128*16, 128*128*32, 128*128*64])
    particle_count4 = np.array([256*256*16, 256*256*32, 256*256*64])

    absolute_error1 = np.array([
        5.722045901990214e-06,
        1.907348629970329e-05,
        5.3405761704539145e-05
    ])
    absolute_error2 = np.array([
        1.1444091800427714e-05,
        2.6702880795426154e-05,
        6.103515630684342e-05
    ])
    absolute_error3 = np.array([
        1.1444091800427714e-05,
        2.6702880902007564e-05,
        6.866455080967171e-05
    ])
    absolute_error4 = np.array([
        1.335144040126579e-05,
        3.433227539773043e-05,
        7.629394541197598e-05
    ])

    error_per_particle1 = absolute_error1 / particle_count1
    error_per_particle2 = absolute_error2 / particle_count2
    error_per_particle3 = absolute_error3 / particle_count3
    error_per_particle4 = absolute_error4 / particle_count4

    plt.plot(np.log2(particle_count1), error_per_particle1, 'o-')
    plt.plot(np.log2(particle_count2), error_per_particle2, 'o-')
    plt.plot(np.log2(particle_count3), error_per_particle3, 'o-')
    plt.plot(np.log2(particle_count4), error_per_particle4, 'o-')

    plt.title('Serial (1 thread, 1 block)')
    plt.xlabel('log2(N)')
    plt.ylabel('Relative error')
    plt.legend(['32x32', '64x64', '128x128', '256x256'], title='Grid cells')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
