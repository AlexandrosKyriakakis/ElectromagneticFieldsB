import matplotlib.pyplot as plt
import numpy as np

d = 2
h = 1
a = 0.1
I = 1
M = I*np.pi*a**2
m_0 = 4*np.pi/10000000


def canvas():
    fig, ax = plt.subplots()
    ax.set_title('Διανυσματικό Δυναμικό Α(x,y)')
    ax.set_xlabel('x(m)')
    ax.set_ylabel('y(m)')
    ax.grid()
    #fig.set_size_inches(2000.5, 2000.5)
    return ax


def Ax(x, y, z):

    f1 = 1/(np.sqrt((x-d)**2 + (y-h)**2 + z**2))
    f2 = 1/(np.sqrt((x-d)**2 + (y+h)**2 + z**2))
    f3 = 1/(np.sqrt((x+d)**2 + (y-h)**2 + z**2))
    f4 = 1/(np.sqrt((x+d)**2 + (y+h)**2 + z**2))
    return (m_0*M/(4*np.pi))*z*(f1**3 - f2**3 + f3**3 - f4**3)


def Ax_aux(x, y):
    return Ax(x, y, 2)


a_x = np.vectorize(Ax_aux)

x = np.linspace(0, 4, 50)
y = np.linspace(0, 4, 50)
X, Y = np.meshgrid(x, y)

Ax = a_x(X, Y)
numrows = len(Ax)
numcols = len(Ax[0])
Ay = np.array([[0 for i in range(numcols)] for i in range(numrows)])

ax = canvas()
ax.streamplot(X, Y, Ax/(2*((Ax**2 + Ay**2)**0.5)),
              Ay/(2*((Ax**2 + Ay**2)**0.5)))
plt.show()
ax = canvas()
ax.quiver(X, Y, Ax/(2*((Ax**2 + Ay**2)**0.5)), Ay/(2*((Ax**2 + Ay**2)**0.5)),
          (Ax**2 + Ay**2)**0.5, cmap='viridis', units='xy', width=0.0035, headwidth=3., headlength=4.)
plt.show()
