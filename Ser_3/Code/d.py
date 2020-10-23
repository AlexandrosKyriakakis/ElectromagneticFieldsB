import matplotlib.pyplot as plt
import numpy as np


a = 0.1
I1 = 1
h = 0.1


def canvas():
    fig, ax = plt.subplots()
    ax.grid()
    #fig.set_size_inches(2000.5, 2000.5)
    return ax


def H(z):
    f1 = 1/((a**2 + (z-h)**2)**1.5)
    f2 = 1/((a**2 + (z+h)**2)**1.5)
    return (I1*a**2/2) * (f1 + f2)


def der_H(z):
    f1 = 1/((a**2 + (z-h)**2)**2.5)
    f2 = 1/((a**2 + (z+h)**2)**2.5)
    return (I1*a**2/2) * (-3)*((z-h)*f1 + (z+h)*f2)


def der2_H(z):
    f1 = 1/((a**2 + (z-h)**2)**3.5)
    f2 = 1/((a**2 + (z+h)**2)**3.5)
    f3 = (a**2 - 4*(z-h)**2)
    f4 = (a**2 - 4*(z+h)**2)
    return (I1*a**2/2) * (-3)*(f3*f1 + f4*f2)


def der3_H(z):
    f1 = 1/((a**2 + (z-h)**2)**4.5)
    f2 = 1/((a**2 + (z+h)**2)**4.5)
    f3 = (3*a**2 - 4*(z-h)**2)
    f4 = (3*a**2 - 4*(z+h)**2)
    return (I1*a**2/2) * 15*((z-h)*f3*f1 + (z+h)*f4*f2)


def der4_H(z):
    f1 = 1/((a**2 + (z-h)**2)**5.5)
    f2 = 1/((a**2 + (z+h)**2)**5.5)
    f3 = (a**4 - 12*a**2*(z-h)**2 + 8*(z-h)**4)
    f4 = (a**4 - 12*a**2*(z+h)**2 + 8*(z+h)**4)
    return (I1*a**2/2) * 45*(f3*f1 + f4*f2)


H = np.vectorize(H)
der_H = np.vectorize(der_H)
der2_H = np.vectorize(der2_H)
der3_H = np.vectorize(der3_H)
der4_H = np.vectorize(der4_H)

z = np.linspace(-2*h, 2*h, 1000)  # 3 φορες για h = 0.025, 0.05, 0.1

ax = canvas()
ax.set(xlabel='z(m)', ylabel="H(z)(A/m)", title='Μαγνητικό Πεδίο Η(z)')
ax.plot(z, H(z))
plt.show()

ax = canvas()
ax.set(xlabel='z(m)', ylabel="H'(z)(A/m^2)",
       title='1η Παράγωγος του Μαγνητικού Πεδίου Η(z)')
ax.plot(z, der_H(z))
plt.show()

ax = canvas()
ax.set(xlabel='z(m)', ylabel="H''(z)(A/m^3)",
       title='2η Παράγωγος του Μαγνητικού Πεδίου Η(z)')
ax.plot(z, der2_H(z))
plt.show()

ax = canvas()
ax.set(xlabel='z(m)', ylabel="H^(3)(z)(A/m^4)",
       title='3η Παράγωγος του Μαγνητικού Πεδίου Η(z)')
ax.plot(z, der3_H(z))
plt.show()

ax = canvas()
ax.set(xlabel='z(m)', ylabel="H^(4)(z)(A/m^5)",
       title='4η Παράγωγος του Μαγνητικού Πεδίου Η(z)')
ax.plot(z, der4_H(z))
plt.show()
