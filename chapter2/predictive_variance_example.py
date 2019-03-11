import numpy as np
import random
import math
import matplotlib.pylab as plt
from numpy.linalg import inv



def main():
    N = 100
    x = np.sort((10*np.random.rand(100,1)-5), axis = None)
    #x = np.sort(x)
    t = 5*np.power(x,3) - np.power(x,2) + x
    noise_var = 300
    t = t + np.random.randn(np.size(x))*math.sqrt(noise_var)
    pos = np.where(np.logical_and(x>0, x<2))[0]
    x = np.delete(x, pos)
    t = np.delete(t, pos)
    testx = np.arange(-5, 5.1, 0.1)
    testx = np.transpose(testx)
    #print(testx)
    #fig = plt.figure()

    for i in range(1, 10):
        X = np.ones((x.size, i), dtype = 'float')
        testX = np.ones((testx.size, i), dtype = 'float')
        for j in range(x.size):
            for k in range(1, i):
                X[j, k] = np.power(x[j], k)

        for j in range(testx.size):
            for k in range(1, i):
                testX[j, k] = np.power(testx[j], k)
        w = np.matmul(np.matmul(inv(np.matmul(np.transpose(X), X)), np.transpose(X)), t)
        testmean = np.matmul(testX, w)
        mean = np.matmul(X, w)
        inverse = inv(np.matmul(np.transpose(X), X))
        var = (1/N)*((np.matmul(np.transpose(t),t)) - (np.matmul(np.transpose(t), mean)))
        testvar = var*np.diag(np.matmul(testX, np.matmul(inverse, np.transpose(testX))))
        #ax = fig.add_subplot(3,3,i)
        #ax.plot(x, t, 'k.')
        plt.plot(x,t,'k.')
        plt.errorbar(testx, testmean, yerr=testvar, fmt = '--.')
        plt.show()
        #ax.errorbar(testx, testmean, yerr=testvar, fmt = 'b.')
        #log_like[i] = np.sum(np.log(multivariate_normal.pdf(tn, mean = mean, cov = math.sqrt(var))), dtype = 'float')
        #print(testvar)
    
    #plt.plot(x, t, 'b.')
    #plt.show()



if __name__ == '__main__':
    main()
