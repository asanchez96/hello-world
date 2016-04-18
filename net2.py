#! /usr/bin/env python
"""

"""

from __future__ import division, print_function
from math import tanh, exp
import numpy as np
import matplotlib.pyplot as plt

from timeit import default_timer as timer

from numbapro import guvectorize

class Net():

    def __init__(self, shape):
        """

        """
        self.shape = shape
        self.num_layers = len(shape)

        num_w = 0
        for i in range(self.num_layers - 1):
            num_w += shape[i+1] + shape[i+1]*shape[i]

        self.weights = np.random.normal(scale=0.1, size=num_w)

    def output(self, x, weights=None):
        """
        ASSUME NEURAL NETWORK ALWAYS HAS 1 OUTPUT
        """
        if weights is None:
            weights = self.weights
        y = x
        w = weights

        for n in range(self.num_layers - 1):

            a = w[:(self.shape[n+1]*self.shape[n])]
            a = np.reshape(a, (self.shape[n+1], self.shape[n]))
            w = w[(self.shape[n+1]*self.shape[n]):]

            b = w[:(self.shape[n+1])]
            w = w[self.shape[n+1]:]

            y = np.array([i + np.dot(j, y) for i, j in zip(b,a)])

            if n < self.num_layers - 1:
                y = [tanh(i) for i in y]

        return y[0]

    def V_i(self, t_i, x_i, weights):
        return (t_i - self.output(x_i, weights))**2


    def V(self, t, x, weights=None):
        """

        """
        if weights is None:
            weights = self.weights

        sigma = 0.1
        v = 0
        for i in range(len(t)):
            v += self.V_i(t[i], x[i], weights)

        return v/(2*sigma**2)

    def dV_dw(self, i, t, x):
        """

        """
        h = 1e-6

        w_m = np.array(self.weights)
        w_p = np.array(self.weights)
        w_m[i] -= h
        w_p[i] += h

        derivative = (self.V(t,x,weights=w_p) - self.V(t,x,weights=w_m))/(2*h)

        return derivative

    def HMC(self, eps, L, M ,t, x):
        """

        """
        w0 = np.array(self.weights)
        #start = timer()

        for m in range(M):
            for i in range(len(self.weights)):

                p = p0 = np.random.normal(scale=0.5)

                for l in range(L):
                    p -= (eps/2)*self.dV_dw(i,t,x)
                    self.weights[i] += (eps)*p
                    p -= (eps/2)*self.dV_dw(i,t,x)

                alpha = exp(p0**2/2 + self.V(t,x,weights=w0)
                -p**2/2 - self.V(t,x))

                if np.random.uniform() <= alpha:
                    w0 = self.weights
                else:
                    self.weights = w0
        #print('t=',timer() - start)


#End class
"""
def sigmoid(f):
    """

    """
    return 1/(1+exp(-f))

"""
"""
n = Net([2,10,1])
n.weights = [ 0.67599493,  0.19253889, -0.46702538,  0.36529877, -0.2231968,  -0.7712298,
 -0.01862811, -0.09478067,  0.08007802,  0.09802791, -0.83054982, -0.81050179,
  0.35720011, -1.16839228, -0.35631495,  0.16123073, -0.10266976, -0.3210038,
 -0.53596741,  0.73367764, -0.05197441, -0.29398686,  0.27636347, -0.38950989,
  0.08008727,  0.31594729,  0.63546336, -0.35329879,  0.41472282,  0.73188751,
 -0.10564109, -0.82210652, -0.18454532, -0.24262225,  0.04661533, -0.1751489,
  0.11801942,  0.53217995, -0.09270765,  0.03782397,  0.21406475]

training_data = np.loadtxt("train.dat")

x,t = [],[]
for i in range(len(training_data)):
    x.append(training_data[i][:2])
    t.append(training_data[i][2])

    x = x[:20]
    t = t[:20]

for i in range(len(x)):
    print(n.output(x[i]))

"""
"""
n = Net([2,10,1])
training_data = np.loadtxt("train.dat")

x,t = [],[]
for i in range(len(training_data)):
    x.append(training_data[i][:2])
    t.append(training_data[i][2])

    x = x[:5]
    t = t[:5]

#w = []
res = []
for i in range(200):
    n.HMC(.001, 50, 20, t, x)
    y = n.output(x[0])
    print(y)
    res.append(y)
    #w.append(n.weights)
print(n.weights)
plt.hist(res[50:], normed=True)
plt.show()
#print(w)
"""
