'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++
++Code to test the over fitting of different equations to
++as close as possible to the data. Fields can be commented in
++or commented out to ptroduce different equations.
++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

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

    X = np.ones((27, 9), dtype = 'float')
    a = 2660
    b = 4.3
    for i in range(27):
        X[i, 1] = xn[i]
        X[i, 2] = xn[i]*xn[i] #np.sin((xn[i]-a)/b)
        X[i, 3] = xn[i]*xn[i] *xn[i]
        X[i, 4] = xn[i]*xn[i] *xn[i]*xn[i]
        X[i, 5] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]
        X[i, 6] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]*xn[i]
        X[i, 7] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]*xn[i]*xn[i]
        X[i, 8] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]*xn[i]*xn[i]*xn[i]

    #print(X)
    XT = np.transpose(X)
    Y = np.matmul(XT, X)
    Y_inv = inv(Y)
    Z = np.matmul(Y_inv, XT)
    W = np.matmul(Z, tn)
    prod = np.matmul(X, W)
    diff = tn - prod
    diff_trans = np.transpose(diff)
    Sq_Loss = np.matmul(diff_trans, diff)
    SqLossAvg = Sq_Loss/27
    #print(W)
    #eq2 = np.zeros(27, dtype='float')
    #for i in range(27):
    #eq4 = W[0]+(W[1]*xn)+(W[2]*xn*xn)+(W[3]*xn*xn*xn)+(W[4]*xn*xn*xn*xn)#+(W[5]*xn*xn*xn*xn*xn)+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    eq8 = W[0]+(W[1]*xn)+(W[2]*xn*xn)+(W[3]*xn*xn*xn)+(W[4]*xn*xn*xn*xn)+(W[5]*xn*xn*xn*xn*xn)+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    #eq2 = W[0] + (W[1]*xn) + (W[2]*xn*xn) + (W[3]*xn*xn*xn) #+ (W[4]*xn*xn*xn*xn)#+ (W[2]*np.sin((xn-a)/b))
    print(eq8)
    #print(tn)
    '''xntn = xn*tn
    xn2 = np.power(xn,2)
    xn_avg = np.average(xn)
    tn_avg = np.average(tn)
    xntn_avg = np.average(xntn)
    xn2_avg = np.average(xn2)
    w1 = (xntn_avg - (xn_avg * tn_avg))/(xn2_avg - (xn_avg*xn_avg))
    w0 = tn_avg - (w1 * xn_avg)
    eq = w0 + (w1 * xn)
    x_new = np.array([2012, 2016, 2020])
    t_new = w0 + (w1 * x_new)
    #print(w1)
    #print(w0)
    #print(t_new)'''
    plt.plot(xn, tn, 'r.')
    #plt.plot(xn, eq, 'b--')
    #plt.plot(x_new, t_new, 'g*')
    plt.plot(xn, eq8, 'k-')
    plt.show()



if __name__ == '__main__':
    main()
