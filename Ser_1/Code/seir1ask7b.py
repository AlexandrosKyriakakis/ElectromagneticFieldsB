import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from mpl_toolkits import mplot3d


def f(x, z):
    k_squ = (4*abs(x))/((abs(x) + 1)**2 + z**2)
    def K(TH): return 1/(np.sqrt(1 - k_squ*((np.sin(TH))**2)))
    kk, error1 = integrate.quad(K, 0, np.pi/2)
    Fi = (2*kk)/(np.pi*np.sqrt((abs(x) + 1)**2 + z**2))
    return Fi


f1 = np.vectorize(f)

x = np.linspace(-10, 10, 100)
z = np.linspace(-10, 10, 100)

X, Y = np.meshgrid(x, z)
Z = f1(X, Y)

fig, ax = plt.subplots()
CS = ax.contour(X, Y, Z, zdir='xy', offset=11,
                levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                        0.7, 0.8, 0.9, 1.0, 1.1, 1.2], cmap=matplotlib.cm.magma,
                color='black')
ax.clabel(CS, CS.levels, inline=True, fontsize=5)
ax.set_title('Iσοδυναμικές επιφάνειες')
ax.set_xlabel('X(m)')
ax.set_ylabel('Z(m)')
plt.show()
