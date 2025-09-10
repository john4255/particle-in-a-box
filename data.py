import numpy as np

n = 3
h = 6.626E-34
m = 9.109E-31
L = 10.0E-9
E = (n * h / L) ** 2 / (8 * m)

def psi_soln(x):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def gen_data(sz=1000):
    ds = np.zeros([sz,2])
    for i in range(sz):
        x = np.random.rand() * L
        psi = psi_soln(x)
        psi_noised = psi + (np.random.rand() * 0.02 - 0.01) * psi
        ds[i] = [x, psi_noised]
    return ds