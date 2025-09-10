import tensorflow as tf
from keras.layers import Dense, Flatten, Conv2D
from keras import Model

import numpy as np
import matplotlib.pyplot as plt

n = 3
h = 6.626E-34
m = 9.109E-31
L = 10.0E-9
E = (n * h / L) ** 2 / (8 * m)

def psi_soln(x):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def gen_data(sz=100):
    ds = np.zeros([sz,2])
    for i in range(sz):
        x = np.random.rand() * L
        if (x > 0.2 * L and x < 0.3 * L) or (x > 0.85 * L):
            continue
        psi = psi_soln(x)
        psi_noised = psi + (np.random.rand() * 0.02 - 0.01) * psi
        ds[i] = [x, psi_noised]
    return ds

ds = gen_data()

fig, ax = plt.subplots()

ax.scatter(ds[:,0], ds[:, 1], c='magenta', marker='x')
ax.set_xticks(np.linspace(0.0, L, 5))
    
plt.show()