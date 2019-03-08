import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import random

def main():
    xn = np.array([1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932,
                   1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980,
                   1984, 1988, 1992, 1996, 2000, 2004, 2008], dtype='float')
    tn = np.array([12, 11, 11, 11.2, 10.8, 10.8, 10.8, 10.6, 10.8, 10.3, 10.3,
                   10.3, 10.4, 10.5, 10.2, 10, 9.95, 10.14, 10.06, 10.25, 9.99,
                   9.92, 9.96, 9.84, 9.87, 9.85, 9.69], dtype='float')

    w = np.array([36.4165, -0.0133])
    w = np.transpose(w)
    X = np.ones((27,2),dtype = 'float')
    for i in range(27):
        X[i,1] = xn[i]

    mean_t = np.matmul(X, w)
    noise_var = 0.01
    noisy_t = mean_t + np.random.randn(np.size(mean_t))*noise_var
    #print(noisy_t)
    plt.plot(xn, mean_t,'r.')
    plt.plot(xn, noisy_t, 'b.')
    plt.plot(xn, tn,'k.')
    plt.show()
    
    





if __name__ == '__main__':
    main()
