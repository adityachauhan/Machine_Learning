'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++
++ Code to test the cross validation techique. Code can be 
++ editied to getboth Leave one out Cross Validation and
++ K-F0ld Cross Validation as per use.
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

    xn_train = xn[0:20]
    xn_valid = xn[20:]
    tn_train = tn[0:20]
    tn_valid = tn[20:]
    
    print(xn)
    print(xn_train)
    print(xn_valid)

    
    X = np.ones((20, 5), dtype = 'float')
    
    for i in range(20):
        X[i, 1] = np.power(xn_train[i],1)
        X[i, 2] = np.power(xn_train[i],2) #*xn_train[i] #np.sin((xn[i]-a)/b)
        X[i, 3] = np.power(xn_train[i],3) #*xn_train[i] *xn_train[i]
        X[i, 4] = np.power(xn_train[i],4) #*xn_train[i] *xn_train[i]*xn_train[i]
        #X[i, 5] = np.power(xn_train[i],5) #*xn_train[i] *xn_train[i]*xn_train[i]*xn_train[i]
        #X[i, 6] = np.power(xn_train[i],6) #*xn_train[i] *xn_train[i]*xn_train[i]*xn_train[i]*xn_train[i]
        #X[i, 7] = np.power(xn_train[i],7) #*xn_train[i] *xn_train[i]*xn_train[i]*xn_train[i]*xn_train[i]*xn_train[i]
        #X[i, 8] = np.power(xn_train[i],8) #*xn_train[i] *xn_train[i]*xn_train[i]*xn_train[i]*xn_train[i]*xn_train[i]*xn_train[i]
    XT = np.transpose(X)
    Y = np.matmul(XT, X)
    Y_inv = inv(Y)
    Z = np.matmul(Y_inv, XT)
    W = np.matmul(Z, tn_train)
    prod = np.matmul(X, W)
    diff = tn_train - prod
    diff_trans = np.transpose(diff)
    Sq_Loss = np.matmul(diff_trans, diff)
    SqLossAvg = Sq_Loss/27
    eq_train = W[0]+(W[1]*xn_train)+(W[2]*np.power(xn_train,2))+(W[3]*np.power(xn_train,3))+(W[4]*np.power(xn_train,4))#+(W[5]*np.power(xn_train,5))+(W[6]*np.power(xn_train,6))+(W[7]*np.power(xn_train,7))+(W[8]*np.power(xn_train,8))
    #print(eq_train)
    eq_valid = W[0]+(W[1]*xn_valid)+(W[2]*np.power(xn_valid,2))+(W[3]*np.power(xn_valid,3))+(W[4]*np.power(xn_valid,4))#+(W[5]*np.power(xn_valid,5))+(W[6]*np.power(xn_valid,6))+(W[7]*np.power(xn_valid,7))+(W[8]*np.power(xn_valid,8))
    #print(eq_valid)
    
    plt.plot(xn, tn, 'r.')
    plt.plot(xn_train, eq_train, 'k-')
    plt.plot(xn_valid, eq_valid, 'b*')
    plt.show()



if __name__ == '__main__':
    main()
