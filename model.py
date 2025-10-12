import tensorflow as tf
from keras.layers import Conv1D, Dense, Dropout
from keras import Model, regularizers

import numpy as np
import matplotlib.pyplot as plt

n = 5 # Energy level
h = 6.626E-19 # Planck's constant (um^2 g / s)
hbar = h / (2.0 * np.pi) # reduced Planck's constant
m = 9.109E-28 # electron mass (g)
L = 10.0E-3 # box size (um)
E = (n * h / L) ** 2 / (8 * m) # Energy (um^2 g / s^2)

@tf.function
def V(x):
    res = tf.math.logical_or(x < 0, x > L)
    return 1.0E10 * tf.cast(res, dtype=tf.float32)

def psi_soln(x):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def gen_data(sz=5000):
    ds = np.zeros([sz,2])
    # sum = 0
    for i in range(sz):
        x = np.random.rand() * L
        if (x > 0.2 * L and x < 0.3 * L) or (x > 0.85 * L):
            continue
        psi = psi_soln(x)
        psi_noised = psi + (np.random.rand() * 0.2 - 0.1) * psi
        ds[i] = [x, psi_noised ** 2]
    # ds[:, 1] /= np.sum(ds[:, 1])
    return ds

class QMModel(Model):
    def __init__(self):
        super().__init__()

        # def E_init(shape, dtype):
        #     return [1.0E-6]
        
        # self.E = self.add_weight(
        #     shape=([1]),
        #     initializer=E_init,
        #     trainable=True,
        # )
        self.d1 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.05))
        self.d2 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.05))
        self.d3 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.05))
        self.d4 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.05)) # TODO: tweak

        self.drop1 = Dropout(0.3)
        self.drop2 = Dropout(0.3)
        self.drop3 = Dropout(0.3)
        self.drop4 = Dropout(0.3)

        self.dout = Dense(1, activation='linear')
    
    def call(self, x):
        x = x / L
        x = self.d1(x)
        x = self.drop1(x)
        x = self.d2(x)
        x = self.drop2(x)
        x = self.d3(x)
        x = self.drop3(x)
        x = self.d4(x)
        x = self.drop4(x)
        wave_func = self.dout(x)

        # Square wavefunction to obtain a probability density value
        predictions = wave_func ** 2

        return predictions, wave_func

model = QMModel()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1.0E-4, clipvalue=10.0)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# Loss parameters (adjust as needed)
data_weight = tf.constant(1.0)
physics_weight = tf.constant(10.0)
# reverse_l2_weight = tf.constant(1.0E3) # turn off
physics_reps = 20

@tf.function
def calc_physics_loss(x, psi, d2psi_dx2):
    p_loss = E * psi + ((hbar ** 2 / (2 * m)) * tf.cast(d2psi_dx2, dtype=tf.float32)) - V(x) * psi
    return p_loss ** 2

@tf.function
def train_step(x, density):
    with tf.GradientTape() as tape:
        predictions, psi = model(x, training=True)
        dpsi_dx = tf.gradients(psi, x)
        d2psi_dx2 = tf.gradients(dpsi_dx, x)
        physics_loss = calc_physics_loss(x, predictions, d2psi_dx2)
        data_loss = loss_object(density, predictions)
        loss = data_weight * data_loss + physics_weight * physics_loss
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def train_physics_step(x):
    with tf.GradientTape() as tape:
        _, psi = model(x, training=True)
        dpsi_dx = tf.gradients(psi, x)
        d2psi_dx2 = tf.gradients(dpsi_dx, x)
        loss = physics_weight * calc_physics_loss(x, psi, d2psi_dx2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def test_step(x, density):
    predictions, psi = model(x, training=False)
    dpsi_dx = tf.gradients(psi, x)
    d2psi_dx2 = tf.gradients(dpsi_dx, x)
    physics_loss = calc_physics_loss(x, predictions, d2psi_dx2)
    data_loss = loss_object(density, predictions)
    t_loss = data_weight * data_loss + physics_weight * physics_loss
    test_loss(t_loss)

EPOCHS = 2000
batch_sz = 64
ds = gen_data()

# Train model
for epoch in range(EPOCHS):
    # Shuffle dataset
    np.random.shuffle(ds)
    train_sz = int(0.9 * len(ds))
    train_ds = np.array(ds[:train_sz])
    test_ds = np.array(ds[train_sz:])

    train_ds = tf.data.Dataset.from_tensor_slices((train_ds[:, 0], train_ds[:, 1])).batch(batch_sz)
    test_ds = tf.data.Dataset.from_tensor_slices((test_ds[:, 0], test_ds[:, 1])).batch(batch_sz)

    # Train data
    train_loss.reset_state()
    for x, psi in train_ds:
        x = tf.reshape([x], shape=(len(x),1,))
        psi = tf.reshape([psi], shape=(len(x),1,))
        train_step(x, psi)
    reg_training_loss = train_loss.result()

    train_loss.reset_state()
    for _ in range(physics_reps):
        x_sample = np.linspace(-0.10 * L, 1.10 * L, 512)
        np.random.shuffle(x_sample)
        for j in range(0, len(x_sample), batch_sz):
            x = x_sample[j:j+batch_sz]
            x = tf.reshape([x], shape=(len(x),1,))
            train_physics_step(x)
    physics_training_loss = train_loss.result()

    # Test data
    test_loss.reset_state()
    for x, psi in test_ds:
        x = tf.reshape([x], shape=(len(x),1,))
        test_step(x, psi)
    reg_test_loss = test_loss.result()

    print(
        f'Epoch {epoch+1:04d}: '
        f'Reg Loss: {reg_training_loss:0.2f}, '
        f'Pure Physics Loss: {physics_training_loss:0.2f}, '
        f'Test Loss: {reg_test_loss:0.2f}, '
    )

model.save(f'model_n={n}.keras')

# print()
# print('=== Training Complete ===')
# print(f'E (real)    = {E}')
# print(f'E (learned) = {model.E.numpy()[0]}')

fig, ax = plt.subplots()

ax.scatter(ds[:,0], ds[:, 1], c='magenta', marker='x', label='Training data')
ax.set_xticks(np.linspace(0.0, L, 5))

x_vis = np.linspace(0.0, L, 250)
psi_vis = np.zeros(250)
probs_vis = np.zeros(250)
for i, x in enumerate(x_vis):
    x = tf.expand_dims([x], axis=0)
    probs, psi = model(x)
    probs_vis[i] = probs
    psi_vis[i] = psi
plt.plot(x_vis, psi_vis, c='lime', label='Pseudo-Wavefunction')
plt.plot(x_vis, probs_vis, c='cyan', label='Probability dist.')

plt.legend(loc='upper right')
plt.savefig(f'solution_n={n}.png')
plt.show()