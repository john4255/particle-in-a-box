import tensorflow as tf
from keras.layers import Dense, Dropout
from keras import Model

import numpy as np
import matplotlib.pyplot as plt

n = 3 # Energy level
h = 6.626E-19 # Planck's constant (um^2 g / s)
hbar = h / (2.0 * np.pi) # reduced Planck's constant
m = 9.109E-28 # electron mass (g)
L = 10.0E-3 # box size (um)
E = (n * h / L) ** 2 / (8 * m) # Energy (um^2 g / s^2)

@tf.function
def V(x):
    if x == 0 or x == L:
        return 1.0E10
    return 0.0

def psi_soln(x):
    return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)

def gen_data(sz=2000):
    ds = np.zeros([sz,2])
    for i in range(sz):
        x = np.random.rand() * L
        if (x > 0.2 * L and x < 0.3 * L) or (x > 0.85 * L):
            continue
        psi = psi_soln(x)
        psi_noised = psi + (np.random.rand() * 0.2 - 0.1) * psi
        ds[i] = [x, psi_noised ** 2]
    return ds

class QMModel(Model):
    def __init__(self):
        super().__init__()

        def E_init(shape, dtype):
            return [1.0E-6]
        
        self.E = self.add_weight(
            shape=([1]),
            initializer=E_init,
            trainable=True,
        )
        self.d1 = Dense(128, activation='gelu')
        self.d2 = Dense(128, activation='gelu')
        self.d3 = Dense(128, activation='gelu')
        self.d4 = Dense(1, activation='linear')

        self.drop1 = Dropout(0.1)
        self.drop2 = Dropout(0.1)

    def call(self, x):
        x = x / L
        x = self.d1(x)
        x = self.drop1(x)
        x = self.d2(x)
        x = self.drop2(x)
        x = self.d3(x)
        x = self.d4(x)
        return x

model = QMModel()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=1.0E-5, clipvalue=0.1)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

data_weight = tf.constant(1.0E1)
physics_weight = tf.constant(1.0E-6)
physics_reps = 5

@tf.function
def calc_physics_loss(x, predictions, d2psi_dx2):
    p_loss = model.E * predictions + (hbar ** 2 / (2 * m) - V(x)) * tf.cast(d2psi_dx2[0], dtype=tf.float32)
    return tf.norm(p_loss)

@tf.function
def train_step(x, psi):
    x = tf.reshape([x], shape=(1,1,))
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        dpsi_dx = tf.gradients(predictions, x)
        d2psi_dx2 = tf.gradients(dpsi_dx, x)
        physics_loss = calc_physics_loss(x, predictions, d2psi_dx2)
        data_loss = loss_object(psi, predictions ** 2)
        loss = data_weight * data_loss + physics_weight * physics_loss
    
    # print(model.trainable_variables)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def train_physics_step(x):
    x = tf.reshape([x], shape=(1,1,))
    with tf.GradientTape() as tape:
        predictions = model(x, training=True)
        dpsi_dx = tf.gradients(predictions, x)
        d2psi_dx2 = tf.gradients(dpsi_dx, x)
        physics_loss = calc_physics_loss(x, predictions, d2psi_dx2)
        loss = physics_weight * physics_loss

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def test_step(x, psi):
    x = tf.reshape([x], shape=(1,1,))
    predictions = model(x, training=False)
    dpsi_dx = tf.gradients(predictions, x)
    d2psi_dx2 = tf.gradients(dpsi_dx, x)
    physics_loss = calc_physics_loss(x, predictions, d2psi_dx2)
    data_loss = loss_object(psi, predictions ** 2)
    t_loss = data_weight * data_loss + physics_weight * physics_loss
    test_loss(t_loss)

EPOCHS = 1000
ds = gen_data()

for epoch in range(EPOCHS):
    # Shuffle dataset
    np.random.shuffle(ds)
    train_sz = int(0.9 * len(ds))
    train_ds = np.array(ds[:train_sz])
    test_ds = np.array(ds[train_sz:])

    train_ds = tf.data.Dataset.from_tensor_slices((train_ds[:, 0], train_ds[:, 1]))
    test_ds = tf.data.Dataset.from_tensor_slices((test_ds[:, 0], test_ds[:, 1]))

    # Train data
    train_loss.reset_state()
    for x, psi in train_ds:
        train_step(x, psi)
    reg_training_loss = train_loss.result()

    train_loss.reset_state()
    for _ in range(physics_reps):
        x_sample = np.linspace(0.0, L, 250)
        for x in x_sample:
            train_physics_step(x)
    physics_training_loss = train_loss.result()

    test_loss.reset_state()
    # Test data
    for x, psi in test_ds:
        test_step(x, psi)
    reg_test_loss = test_loss.result()

    print(
        f'Epoch {epoch+1:04d}: '
        f'Reg Loss: {reg_training_loss:0.2f}, '
        f'Pure Physics Loss: {physics_training_loss:0.2f}, '
        f'Test Loss: {reg_test_loss:0.2f}, '
    )

model.save('model.keras')

print()
print('=== Training Complete ===')
print(f'E (real)    = {E}')
print(f'E (learned) = {model.E.numpy()[0]}')

fig, ax = plt.subplots()

ax.scatter(ds[:,0], ds[:, 1], c='magenta', marker='x')
ax.set_xticks(np.linspace(0.0, L, 5))

x_vis = np.linspace(0.0, L, 250)
psi_vis = np.zeros(250)
probs_vis = np.zeros(250)
for i, x in enumerate(x_vis):
    x = tf.reshape([x], shape=(1,1))
    psi = model(x)
    psi_vis[i] = psi
    probs_vis[i] = psi ** 2
plt.plot(x_vis, psi_vis, c='lime')
plt.plot(x_vis, probs_vis, c='cyan')

plt.savefig('solution.png')
plt.show()