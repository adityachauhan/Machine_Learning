import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import multivariate_normal
import math

def main():
    N = 27
    xn = np.array([1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932,
                   1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980,
                   1984, 1988, 1992, 1996, 2000, 2004, 2008], dtype='float')
    tn = np.array([12, 11, 11, 11.2, 10.8, 10.8, 10.8, 10.6, 10.8, 10.3, 10.3,
                   10.3, 10.4, 10.5, 10.2, 10, 9.95, 10.14, 10.06, 10.25, 9.99,
                   9.92, 9.96, 9.84, 9.87, 9.85, 9.69], dtype='float')

    xn = xn - xn[0]
    xn = xn/4
    log_like = np.zeros((10), dtype='float')
    orders = np.arange(10)
    for i in range(1, 10):
        X = np.ones((27, i), dtype = 'float')
        for j in range(27):
            for k in range(1, i):
                X[j, k] = np.power(xn[j], k)
        w = np.matmul(np.matmul(inv(np.matmul(np.transpose(X), X)), np.transpose(X)), tn)
        mean = np.matmul(X, w)
        var = (1/N)*((np.matmul(np.transpose(tn),tn)) - (np.matmul(np.transpose(tn), mean)))
        log_like[i] = np.sum(np.log(multivariate_normal.pdf(tn, mean = mean, cov = math.sqrt(var))), dtype = 'float')
        #print(var)
        #print(X.size)
    #print(log_like)
    plt.plot(orders, log_like, 'b-')
    plt.show()



if __name__ == '__main__':
    main()
