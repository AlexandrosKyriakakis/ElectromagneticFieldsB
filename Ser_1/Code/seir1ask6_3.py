import scipy as sp
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from mpl_toolkits import mplot3d


def f(r, TH1):
    x = r*np.cos(TH1)
    z = r*np.sin(TH1)

    def Ex1(TH): return (x*(np.sqrt(x**2+z**2) - 2*np.cos(TH1 - TH)) - 2*z *
                         np.sin(TH1 - TH))/(x**2 + z**2 + 4 - 4*np.sqrt(x**2+z**2)*np.cos(TH1 - TH))**1.5
    a, error1 = integrate.quad(Ex1, -0.75, 0.75)

    def Ex2(TH): return (x*(np.sqrt(x**2+z**2) - 0.5*np.cos(TH1 - TH)) - 0.5*z *
                         np.sin(TH1 - TH))/(x**2 + z**2 + 0.25 - np.sqrt(x**2+z**2)*np.cos(TH1 - TH))**1.5
    b, error2 = integrate.quad(Ex2, -0.75, 0.75)

    def Ey1(TH): return (z*(np.sqrt(x**2+z**2) - 2*np.cos(TH1 - TH)) - 2*x *
                         np.sin(TH1 - TH))/(x**2 + z**2 + 4 - 4*np.sqrt(x**2+z**2)*np.cos(TH1 - TH))**1.5
    c, error3 = integrate.quad(Ey1, -0.75, 0.75)

    def Ey2(TH): return (z*(np.sqrt(x**2+z**2) - 0.5*np.cos(TH1 - TH)) - 0.5*x *
                         np.sin(TH1 - TH))/(x**2 + z**2 + 0.25 - np.sqrt(x**2+z**2)*np.cos(TH1 - TH))**1.5
    d, error4 = integrate.quad(Ey2, -0.75, 0.75)

    Ex = ((2)/(np.sqrt(x**2 + z**2)))*a - (1/(np.sqrt(x**2 + z**2)))*b
    Ez = ((2)/(np.sqrt(x**2 + z**2)))*c - (1/(np.sqrt(x**2 + z**2)))*d

    return Ex, Ez


f1 = np.vectorize(f)
r = np.linspace(1, 8, 100)
TH1 = np.linspace(0, 2*np.pi, 100)
R, TH = np.meshgrid(r, TH1)
Ex, Ez = f1(R, TH)

fig, ax = plt.subplots()

ax.quiver(R*np.cos(TH), R*np.sin(TH), Ex/(Ex**2+Ez**2)**0.5, Ez/(Ex**2+Ez**2)**0.5, (Ex**2+Ez**2)
          ** 0.5, cmap=matplotlib.cm.plasma, units='xy', scale=8, zorder=3, width=0.015, headwidth=3., headlength=4.)

ax.set_title('Δυναμικές Γραμμές Ηλεκτρικού Πεδίου')
ax.set_xlabel('X(m)')
ax.set_ylabel('Z(m)')
plt.show()
