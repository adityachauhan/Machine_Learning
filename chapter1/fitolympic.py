'''
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++
++ Code of basic model based on Linear equation
++ derived from the olympic data for 100m men sprint
++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
'''
import numpy as np
import matplotlib.pyplot as plt

def main():
    N = 27
    xn = np.array([1896, 1900, 1904, 1906, 1908, 1912, 1920, 1924, 1928, 1932,
                   1936, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980,
                   1984, 1988, 1992, 1996, 2000, 2004, 2008], dtype='float')
    tn = np.array([12, 11, 11, 11.2, 10.8, 10.8, 10.8, 10.6, 10.8, 10.3, 10.3,
                   10.3, 10.4, 10.5, 10.2, 10, 9.95, 10.14, 10.06, 10.25, 9.99,
                   9.92, 9.96, 9.84, 9.87, 9.85, 9.69], dtype='float')
    xntn = xn*tn
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
    print(w1)
    print(w0)
    print(t_new)
    plt.plot(xn, tn, 'r.')
    plt.plot(xn, eq, 'b-')
    plt.plot(x_new, t_new, 'g*')
    plt.show()



if __name__ == '__main__':
    main()
