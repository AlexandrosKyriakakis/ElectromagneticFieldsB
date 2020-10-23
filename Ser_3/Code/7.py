import matplotlib.pyplot as plt
import autograd.numpy as np
from autograd import grad

a = 0.025
B = 0.25
C = 0.12
b = B/2
c = C/2
h = 0.2
m0 = 4*np.pi/10000000
N = 100
I1 = 20
f = 30


def l(th):
    f1 = 1/(np.sqrt(h**2 + b**2 + c**2 + 2*c*h*np.sin(th)))
    f2 = 1/(np.sqrt(h**2 + b**2 + c**2 - 2*c*h*np.sin(th)))
    f3 = 1/(h**2 + c**2 + 2*c*h*np.sin(th))
    f4 = 1/(h**2 + c**2 - 2*c*h*np.sin(th))
    f5 = (2*b*np.cos(th)/(b**2 + h**2*(np.cos(th))**2))
    return (m0*a**2/4) * (f5*((h*np.sin(th) + c)*f1 - (h*np.sin(th) - c)*f2) +
                          (2*b*c*np.cos(th)*f1*f3) + (2*b*c*np.cos(th)*f2*f4))


L = np.vectorize(l)
DerL = np.vectorize(grad(l))


def th(t):
    return 2*np.pi*f*t


def aux(t):
    return N*I1*l(th(t))


e = grad(aux)
E = np.vectorize(e)

theta = np.linspace(0, 4*np.pi, 1000)

# plot L
fig, ax = plt.subplots()
ax.set(xlabel='Θ(rad)', ylabel='L(Θ)(H)', title='Αλληλεπαγωγή L(Θ)')
ax.grid()

ax.plot(theta, L(theta))
plt.show()

# plot DerL
fig, ax = plt.subplots()
ax.set(xlabel='Θ(rad)', ylabel="dL(Θ)/dΘ (H/rad)",
       title='Ρυθμός Mεταβολής Αλληλεπαγωγής dL(Θ)/dΘ')
ax.grid()

ax.plot(theta, DerL(theta))
plt.show()

# plot E(f)^2
t = np.linspace(0, 5/f, 1000)
timestep = ((5/f) + 1) / t.size
E_f = np.fft.fft(E(t))
E_f = np.absolute(E_f)
freq = np.fft.fftfreq(t.size, d=timestep)
fig, ax = plt.subplots()
ax.set(xlabel='f(Hz)', ylabel='E(f)^2 ((V*s)^2)',
       title='Φασματική Πυκνότητα Ενέργειας E(f)^2')
ax.grid()

ax.plot(freq, np.power(E_f, 2))
plt.show()

# fourier series coefficients with monte carlo integration, using 10000 samples
monte_time = np.linspace(0, 1/f, 10000)
E_t = E(monte_time)


def an(n):
    a = E_t * np.cos(2*np.pi*n*f*monte_time)
    return (2*a.sum())/a.size


def bn(n):
    b = E_t * np.sin(2*np.pi*n*f*monte_time)
    return (2*b.sum())/b.size


def approx(x, N):
    fourier = np.array([an(i) * np.cos(2*np.pi*i*f*x) + bn(i)
                        * np.sin(2*np.pi*i*f*x) for i in range(1, N+1)])
    return fourier.sum() + (an(0) / 2)


Nh = 10  # pano orio tou athroismatos sthn seira fourier


def fourier(x):
    return approx(x, Nh)


fourier = np.vectorize(fourier)

# plot e(t) and fourier series approximation
t = np.linspace(0, 5/f, 10000)
fig, ax = plt.subplots()
ax.set(xlabel='t(s)', ylabel='e(t)(Volt)', title='Ηλεκτρεργετική Δύναμη e(t)')
ax.grid()

ax.plot(t, E(t), t, fourier(t))
plt.show()

pinakas = [[]]
pinakas[0].append('n')
pinakas[0].append('a_n')
pinakas[0].append('b_n')
print('a0 = ' + str(an(0)))
for i in range(1, 100 + 1):
    pinakas.append([])
    row = pinakas[i]
    row.append(str(i))
    row.append(an(i))
    row.append(bn(i))
pinakas = np.array(pinakas)
print(pinakas)
