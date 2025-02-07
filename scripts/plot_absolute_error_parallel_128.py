import matplotlib.pyplot as plt
import numpy as np

def main():
    particle_count1 = np.array([32*32*16, 32*32*32, 32*32*64])
    particle_count2 = np.array([64*64*16, 64*64*32, 64*64*64])
    particle_count3 = np.array([128*128*16, 128*128*32, 128*128*64])
    particle_count4 = np.array([256*256*16, 256*256*32, 256*256*64])

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

    error_per_particle1 = absolute_error1 / particle_count1
    error_per_particle2 = absolute_error2 / particle_count2
    error_per_particle3 = absolute_error3 / particle_count3
    error_per_particle4 = absolute_error4 / particle_count4

    plt.plot(np.log2(particle_count1), error_per_particle1, 'o-')
    plt.plot(np.log2(particle_count2), error_per_particle2, 'o-')
    plt.plot(np.log2(particle_count3), error_per_particle3, 'o-')
    plt.plot(np.log2(particle_count4), error_per_particle4, 'o-')

    plt.title('parallel (128 threads/block)')
    plt.xlabel('log2(N)')
    plt.ylabel('Relative error')
    plt.legend(['32x32', '64x64', '128x128', '256x256'], title='Grid cells')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
