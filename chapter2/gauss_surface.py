import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal
import math

def main():
    mu = np.array([0,0])
    #mu = np.transpose(mu)
    sigma = np.array([0.5,0.5])
    X,Y = np.mgrid[-1:1:30j, -1:1:30j]
    XY = np.column_stack([X.flat, Y.flat])
    covar = np.diag(sigma**2)
    Z = multivariate_normal.pdf(XY, mean=mu, cov = covar)
    Z = Z.reshape(X.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    ax.plot_surface(X,Y,Z)
    plt.show()
    #print(Z)
    #print(covar)



if __name__ == '__main__':
    main()
