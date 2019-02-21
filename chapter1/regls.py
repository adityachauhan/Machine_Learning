'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++
++ Code to test the effect of regularization parameter on
++ 5th order polynomial equation
++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import random
import math

def main():
    N = 27
##    xn = np.array([1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932,
##                   1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980,
##                   1984, 1988, 1992, 1996, 2000, 2004, 2008], dtype='float')
##    tn = np.array([12, 11, 11, 11.2, 10.8, 10.8, 10.8, 10.6, 10.8, 10.3, 10.3,
##                   10.3, 10.4, 10.5, 10.2, 10, 9.95, 10.14, 10.06, 10.25, 9.99,
##                   9.92, 9.96, 9.84, 9.87, 9.85, 9.69], dtype='float')
##
##    xn = xn - xn[0]
##    xn = xn/4

    xn = np.array([0,0.2,0.4,0.6,0.8,1])
    yn = (2*xn)-3
    tn = yn + math.sqrt(3)*np.random.randn(xn.size)
    print(tn)
    X = np.ones((6, 6), dtype = 'float')
    a = 2660
    b = 4.3
    for i in range(6):
        X[i, 1] = xn[i]
        X[i, 2] = xn[i]*xn[i] #np.sin((xn[i]-a)/b)
        X[i, 3] = xn[i]*xn[i]*xn[i]
        X[i, 4] = xn[i]*xn[i]*xn[i]*xn[i]
        X[i, 5] = xn[i]*xn[i]*xn[i]*xn[i]*xn[i]
        #X[i, 6] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]*xn[i]
        #X[i, 7] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]*xn[i]*xn[i]
        #X[i, 8] = xn[i]*xn[i] *xn[i]*xn[i]*xn[i]*xn[i]*xn[i]*xn[i]

    #print(X)
    XT = np.transpose(X)
    Y = np.matmul(XT, X)
    Lamda1 = 0.000001
    Lamda2 = 0.01
    Lamda3 = 0.1
    Lamda4 = 0
    Lamda5 = 1
    #print(Y)
    Y1 = Y + 5*Lamda1*np.identity(6)
    Y2 = Y + 5*Lamda2*np.identity(6)
    Y3 = Y + 5*Lamda3*np.identity(6)
    Y4 = Y + 5*Lamda4*np.identity(6)
    Y5 = Y + 5*Lamda5*np.identity(6)
    #print(Y)
    Y1_inv = inv(Y1)
    Y2_inv = inv(Y2)
    Y3_inv = inv(Y3)
    Y4_inv = inv(Y4)
    Y5_inv = inv(Y5)
    Z1 = np.matmul(Y1_inv, XT)
    Z2 = np.matmul(Y2_inv, XT)
    Z3 = np.matmul(Y3_inv, XT)
    Z4 = np.matmul(Y4_inv, XT)
    Z5 = np.matmul(Y5_inv, XT)
    W1 = np.matmul(Z1, tn)
    W2 = np.matmul(Z2, tn)
    W3 = np.matmul(Z3, tn)
    W4 = np.matmul(Z4, tn)
    W5 = np.matmul(Z5, tn)
    print(W1)
    #prod = np.matmul(X, W)
    #diff = tn - prod
    #diff_trans = np.transpose(diff)
    #Sq_Loss = np.matmul(diff_trans, diff)
    #SqLossAvg = Sq_Loss/27
    #print(W)
    #eq2 = np.zeros(27, dtype='float')
    #for i in range(27):
    #eq4 = W[0]+(W[1]*xn)+(W[2]*xn*xn)+(W[3]*xn*xn*xn)+(W[4]*xn*xn*xn*xn)#+(W[5]*xn*xn*xn*xn*xn)+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    eq1 = W1[0]+(W1[1]*xn)+(W1[2]*xn*xn)+(W1[3]*xn*xn*xn)+(W1[4]*xn*xn*xn*xn)+(W1[5]*xn*xn*xn*xn*xn)#+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    eq2 = W2[0]+(W2[1]*xn)+(W2[2]*xn*xn)+(W2[3]*xn*xn*xn)+(W2[4]*xn*xn*xn*xn)+(W2[5]*xn*xn*xn*xn*xn)#+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    eq3 = W3[0]+(W3[1]*xn)+(W3[2]*xn*xn)+(W3[3]*xn*xn*xn)+(W3[4]*xn*xn*xn*xn)+(W3[5]*xn*xn*xn*xn*xn)#+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    eq4 = W4[0]+(W4[1]*xn)+(W4[2]*xn*xn)+(W4[3]*xn*xn*xn)+(W4[4]*xn*xn*xn*xn)+(W4[5]*xn*xn*xn*xn*xn)#+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    eq5 = W5[0]+(W5[1]*xn)+(W5[2]*xn*xn)+(W5[3]*xn*xn*xn)+(W5[4]*xn*xn*xn*xn)+(W5[5]*xn*xn*xn*xn*xn)#+(W[6]*xn*xn*xn*xn*xn*xn)+(W[7]*xn*xn*xn*xn*xn*xn*xn)+(W[8]*xn*xn*xn*xn*xn*xn*xn*xn)
    print(eq4)
    #eq2 = W[0] + (W[1]*xn) + (W[2]*xn*xn) + (W[3]*xn*xn*xn) #+ (W[4]*xn*xn*xn*xn)#+ (W[2]*np.sin((xn-a)/b))
    #print(eq8)
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
    plt.plot(xn, eq1, 'r-', label="0.000001")
    plt.plot(xn, eq2, 'b-', label="0.01")
    plt.plot(xn, eq3, 'g-', label="0.1")
    plt.plot(xn, eq4, 'k-', label="0")
    plt.plot(xn, eq5, 'c-', label="1")
    plt.show()



if __name__ == '__main__':
    main()
