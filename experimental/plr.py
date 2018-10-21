from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
length = 20
hour = False
if hour:
    filename = 'dataset_files/training_sets/hourly_indicators.csv'
else:
    filename = 'dataset_files/training_sets/minute_indicators.csv'
if os.path.exists(filename):
    print("Reading in dataset with filename {}".format(filename))
    dataset = pd.read_csv(filename, index_col=0)
index = dataset.index
x = np.arange(length, dtype=float)
# x  = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15], dtype=float)
y = dataset.close.head(len(x)).values
# y = np.array([5, 7, 9, 11, 13, 15, 28.92, 42.81, 56.7, 70.59, 84.47, 98.36, 112.25, 126.14, 140.03])
x1, y1 = x, y

def piecewise_linear(x,x0,x1,y0,y1,k0,k1,k2):
    return np.piecewise(x , [x <= x0, np.logical_and(x0<x, x<= x1),x>x1] , [lambda x:k0*x + y0, lambda x:k1*(x-x0)+y1+k0*x0,
                                                                            lambda x:k2*(x-x1) + y0+y1+k0*x0+k1*(x1-x0)])
# def piecewise_linear(x, x0, y0, k1, k2):
#     return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

perr_min = np.inf
p_best = None

for n in range(500):
    k = np.random.rand(7)
    p , e = optimize.curve_fit(piecewise_linear, x1, y1, p0=k)
    perr = np.sum(np.abs(y1-piecewise_linear(x1, *p)))
    if(perr < perr_min):
        perr_min = perr
        p_best = p


xd = np.linspace(0, length, 100)
plt.plot(x, y, "o")
y_out = piecewise_linear(xd, *p_best)
print(p_best)

plt.plot(xd, y_out)
plt.show()