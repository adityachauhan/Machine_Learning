import numpy as np
import matplotlib.pyplot as plt
import random

def main():
    ys = np.random.random_sample((100000,))
    ey2 = np.mean(ys**2)
    posns = np.arange(0, 100000, 10)
    ey2_evol = np.zeros((posns.size), dtype = 'float')
    for i in range(posns.size):
        ey2_evol[i] = np.mean(ys[0:posns[i]]**2)
    plt.semilogx(posns, ey2_evol)
    plt.semilogx([posns[0], posns[9999]],[ey2, ey2], '--r')
    plt.show()





if __name__ == '__main__':
    main()
