import tensorflow as tf
from keras.layers import Dense, Normalization, Input
from keras import Model

import numpy as np
import matplotlib.pyplot as plt

# Define constants using nanometers, amu
n = 3 # Energy level
h = 6.626E-34 * 1.0E18 * 6.022E26 # Planck's constant (nm^2 amu / s)
m = 0.0005485 # electron mass (amu)
L = 10 # nanometers
E = (n * h / L) ** 2 / (8 * m)

def psi_soln(x):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def gen_data(sz=1000):
    ds = np.zeros([sz,2])
    for i in range(sz):
        x = np.random.rand() * L
        if (x > 0.2 * L and x < 0.3 * L) or (x > 0.85 * L):
            continue
        psi = psi_soln(x)
        psi_noised = psi + (np.random.rand() * 0.2 - 0.1) * psi
        ds[i] = [x, psi_noised]
    return ds

ds = gen_data()

fig, ax = plt.subplots()

# Slice data
train_sz = int(0.9 * len(ds))
train_ds = np.array(ds[:train_sz])
test_ds = np.array(ds[train_sz:])

train_ds = tf.data.Dataset.from_tensor_slices((train_ds[:, 0], train_ds[:, 1]))
test_ds = tf.data.Dataset.from_tensor_slices((test_ds[:, 0], test_ds[:, 1]))

ax.scatter(ds[:,0], ds[:, 1], c='magenta', marker='x')
ax.set_xticks(np.linspace(0.0, L, 5))

class QMModel(Model):
    def __init__(self):
        super().__init__()
        self.d1 = Dense(32, activation='relu')
        self.d2 = Dense(32, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.d4 = Dense(1, activation='linear')

    def call(self, x):
        x = x / L
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

model = QMModel()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1.0E-4)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

@tf.function
def train_step(x, psi):
    x = tf.reshape([x], shape=(1,1,))
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        loss = loss_object(psi, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def test_step(x, psi):
    x = tf.reshape([x], shape=(1,1,))
    predictions = model(x, training=False)
    t_loss = loss_object(psi, predictions)
    test_loss(t_loss)

EPOCHS = 50

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_state()
    test_loss.reset_state()

    for x, psi in train_ds:
        train_step(x, psi)

    for x, psi in test_ds:
        test_step(x, psi)

    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result():0.2f}, '
        f'Test Loss: {test_loss.result():0.2f}, '
    )

x_vis = np.linspace(0.0, L, 100)
psi_vis = np.zeros(100)
for i, x in enumerate(x_vis):
    x = tf.reshape([x], shape=(1,1))
    psi = model(x)
    psi_vis[i] = psi
plt.plot(x_vis, psi_vis, c='lime')

plt.show()