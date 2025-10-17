import tensorflow as tf
from keras.layers import Conv1D, Dense, Dropout
from keras import Model, regularizers

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

n = 5 # Principal quantum number
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
        if (x > 0.2 * L and x < 0.35 * L) or (x > 0.85 * L):
            continue
        p = psi_soln(x) ** 2
        p_noised = p + (np.random.rand() * 0.3 - 0.15) * p
        ds[i] = [x, p_noised]
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
        self.d1 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.01))
        self.d2 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.01))
        self.d3 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.01))
        self.d4 = Dense(128, activation='gelu', kernel_regularizer=regularizers.L2(0.01))

        self.drop1 = Dropout(0.4)
        self.drop2 = Dropout(0.4)
        self.drop3 = Dropout(0.4)
        self.drop4 = Dropout(0.4)

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
optimizer = tf.keras.optimizers.SGD(learning_rate=1.0E-5, clipvalue=100.0)

train_loss = tf.keras.metrics.Mean(name='train_loss')
test_loss = tf.keras.metrics.Mean(name='test_loss')

# Loss parameters (adjust as needed)
data_weight = tf.constant(1.0)
physics_weight = tf.constant(1.0E3) # TODO: test
global_physics_weight = tf.constant(1.0)
normalization_weight = tf.constant(1.0E-6)
bc_weight = tf.constant(1.0E-2)

@tf.function
def calc_physics_loss(x, psi, d2psi_dx2):
    p_loss = E * psi + ((hbar ** 2 / (2 * m)) * tf.cast(d2psi_dx2, dtype=tf.float32)) - V(x) * psi
    return tf.norm(p_loss) ** 2

@tf.function
def train_step(x, density):
    with tf.GradientTape() as tape:
        predictions, psi = model(x, training=True)
        dpsi_dx = tf.gradients(psi, x)
        d2psi_dx2 = tf.gradients(dpsi_dx, x)
        physics_loss = calc_physics_loss(x, psi, d2psi_dx2)
        data_loss = loss_object(density, predictions)
        # tf.print(physics_loss)
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
        loss = global_physics_weight * calc_physics_loss(x, psi, d2psi_dx2)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def train_normalize_step(x):
    with tf.GradientTape() as tape:
        _, psi = model(x, training=True)
        dx = L / 1024
        loss = normalization_weight * ((tf.norm(psi) ** 2) * dx - 1.0) ** 2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)

@tf.function
def train_bc_step(x):
    with tf.GradientTape() as tape:
        _, psi = model(x, training=True)
        loss = bc_weight * tf.norm(psi) ** 2
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

@tf.function
def test_step(x, density):
    predictions, psi = model(x, training=False)
    dpsi_dx = tf.gradients(psi, x)
    d2psi_dx2 = tf.gradients(dpsi_dx, x)
    physics_loss = calc_physics_loss(x, psi, d2psi_dx2)
    data_loss = loss_object(density, predictions)
    t_loss = data_weight * data_loss + physics_weight * physics_loss
    test_loss(t_loss)

EPOCHS = 1000 # Adjust for different scenarios
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


    # Correct global structure
    if epoch % 25 == 0:
        # Normalize wavefunction
        x = np.linspace(0.0, L, 1024)
        x = tf.reshape([x], shape=(1024,1))
        for _ in range(5):
            train_normalize_step(x)

        # Physics scan
        for _ in range(25):
            x_sample = np.linspace(-0.2 * L, 1.2 * L, 1024)
            np.random.shuffle(x_sample)
            for j in range(0, len(x_sample), batch_sz):
                x = x_sample[j:j+batch_sz]
                x = tf.reshape([x], shape=(len(x),1,))
                train_physics_step(x)
    
    # Enforce BC
    # x1 = tf.reshape([0.0], shape=(1,1))
    # x2 = tf.reshape([L], shape=(1,1))
    # train_bc_step(x1)
    # train_bc_step(x2)

    # Test data
    test_loss.reset_state()
    for x, psi in test_ds:
        x = tf.reshape([x], shape=(len(x),1,))
        test_step(x, psi)
    reg_test_loss = test_loss.result()

    print(
        f'Epoch {epoch+1:04d}: '
        f'Loss: {reg_training_loss:0.2f}, '
        # f'Pure Physics Loss: {physics_training_loss:0.2f}, '
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
psi_real = np.zeros(250)
for i, x in enumerate(x_vis):
    x = tf.expand_dims([x], axis=0)
    probs, psi = model(x)
    probs_vis[i] = probs
    psi_vis[i] = psi
    psi_real[i] = psi_soln(x)
plt.plot(x_vis, psi_vis, c='lime', label='Pseudo-Wavefunction')
plt.plot(x_vis, probs_vis, c='cyan', label='Probability dist.')
plt.plot(x_vis, psi_real, c='yellow', label='Real Wavefunction')

plt.legend(loc='upper right')
plt.savefig(f'solution_n={n}.png')
plt.show()