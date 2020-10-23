import scipy as sp
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from mpl_toolkits import mplot3d


def f(x, z):
    rt = abs(x)
    a = 1
    v0 = 1
    pi = np.pi
    k = (2*np.sqrt(rt*a))/(np.sqrt((rt+a)**2 + z**2))

    def K(TH): return 1/(np.sqrt(1 - k**2 * (np.sin(TH))**2))
    kk, error4 = integrate.quad(K, 0, pi/2)

    dk_drt = (np.sqrt(a/rt))*(a**2 + z**2 - rt**2)/(((rt + a)**2 + z**2)**1.5)

    def Ek(TH): return (np.sqrt(1 - (k**2) * (np.sin(TH))**2))
    ek, error5 = integrate.quad(Ek, 0, pi/2)

    dK_dk = (ek - (1-k**2)*kk)/(k*(1-k**2))

    Ert = (((-1)*v0*np.sqrt(a))/pi) * (-0.5*(rt**(-1.5))
                                       * k*kk + (rt**(-0.5))*(kk + k*dK_dk)*dk_drt)

    Ez = ((v0)/(2*pi))*((4*z*a)/((rt+a)**2 + z**2)**1.5)*(kk+k*dK_dk)

    return Ert, Ez


f1 = np.vectorize(f)
x = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)
X, Z = np.meshgrid(x, z)
Ex, Ez = f1(X, Z)

fig, ax = plt.subplots()
ax.quiver(X, Z, Ex/((Ex**2+Ez**2)**0.5), Ez/((Ex**2+Ez**2)**0.5), (Ex**2+Ez**2)**0.5,
          cmap=matplotlib.cm.cividis, units='xy', scale=10, zorder=3, width=0.006, headwidth=3., headlength=4.)
ax.set_title('Δυναμικές Γραμμές Ηλεκτρικού Πεδίου')
ax.set_xlabel('X(m)')
ax.set_ylabel('Z(m)')
plt.show()
