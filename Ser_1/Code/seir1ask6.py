import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate


def f(t):
    def x(theta): return (-6*8.85)/((5-4*np.cos(t - theta))**1.5)
    a, error = integrate.quad(x, -0.75, 0.75)
    return a


f2 = np.vectorize(f)
t = np.arange(0.0, 2*np.pi, 0.001)
fig, ax = plt.subplots()
ax.plot(t, f2(t), color='red')

ax.set(xlabel='θ(rad)', ylabel='σ(θ)',
       title='Επιφανειακή πυκνότητα φορτίου')


fig.savefig("Ser1_Ex6_1.png")
