import scipy as sp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from mpl_toolkits import mplot3d


def f(r, TH1):
    def x(TH): return (2)/np.sqrt(r**2 + 4 - 4*r*np.cos(TH1-TH))
    a, error1 = integrate.quad(x, -0.75, 0.75)

    def y(TH): return (-1)/np.sqrt(r**2 + 0.25 - r*np.cos(TH1-TH))
    b, error2 = integrate.quad(y, -0.75, 0.75)
    return a + b


f1 = np.vectorize(f)
r = np.linspace(1, 8, 100)
TH1 = np.linspace(0, 2*np.pi, 100)

X, Y = np.meshgrid(r, TH1)
Z = f1(X, Y)

fig, ax = plt.subplots()
ax.contourf(X*np.cos(Y), X*np.sin(Y), Z, zdir='xy', offset=11,
            levels=[0.01, 0.025, 0.05, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3,
                    0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0],
            cmap=matplotlib.cm.magma)
CS = ax.contour(X*np.cos(Y), X*np.sin(Y), Z, zdir='xy', offset=11,
                levels=[0.01, 0.025, 0.05, 0.1, 0.15, 0.175, 0.2, 0.25, 0.3,
                        0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0],
                color='black')
ax.clabel(CS, CS.levels, inline=True, fontsize=10)
ax.set_title('Ισοδυναμικές επιφάνειες')
ax.set_xlabel('X(m)')
ax.set_ylabel('Z(m)')
plt.show()
fig.savefig("Ser1_Ex6_2")
