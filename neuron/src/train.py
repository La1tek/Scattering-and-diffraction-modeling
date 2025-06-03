# train.py

import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.model_selection import train_test_split

import tensorflow as tf

from model import build_maxwellnet_unet_3d
from physics import PhysicalProperties, helmholtz_pinn_loss
from visualize import generate_and_visualize

# ------------------------
# 8) Training
# ------------------------

# 1) Load and Prepare Data
data_file = 'train.npy'
RI = np.load(data_file).astype(np.float32)
# Добавляем канал измерения: (N, Ny, Nx, Nz, 1)
RI = RI[..., np.newaxis]

# Выделяем контрольный пример для мониторинга
RI_monitor = RI[-1:,...]  # последний пример, shape=(1,Ny,Nx,Nz,1)

# Делим на train/val
RI_train, RI_val = train_test_split(RI, test_size=0.1, shuffle=False)

# Параметры батчей
batch_size = 5
dataset_train = tf.data.Dataset.from_tensor_slices(RI_train).batch(batch_size)
dataset_val   = tf.data.Dataset.from_tensor_slices(RI_val).batch(batch_size)

# 2) Physical properties
Ny, Nx, Nz = RI.shape[1:4]
wavelength = 1.03  # um
dx = dy = dz = 0.1 # um
n0 = 1.333
phys_props = PhysicalProperties(Ny, Nx, Nz, wavelength, dy=dy, dx=dx, dz=dz, n0=n0)

# 3) Build Model and Optimizer
model = build_maxwellnet_unet_3d(input_shape=(Ny, Nx, Nz, 1),
                                  base_channels=16,
                                  depth=4,
                                  activation='elu',
                                  normalization=0.20)
model.summary()

# learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-5,
    decay_steps=100,
    decay_rate=0.5,
    staircase=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


# 4) Training loop отсюда
@tf.function
def train_step(model, RI_batch, phys_props, optimizer, pad=4):
    """
    Один шаг тренировки: предсказание, вычисление PINN-loss и шаг оптимизатора.
    """
    with tf.GradientTape() as tape:
        U_pred = model(RI_batch, training=True)  # (B,Ny,Nx,Nz,2)
        loss   = helmholtz_pinn_loss(U_pred, RI_batch, phys_props, pad)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def val_step(model, RI_batch, phys_props, pad=4):
    """Один шаг валидации без обновления весов"""
    U_pred = model(RI_batch, training=False)
    loss   = helmholtz_pinn_loss(U_pred, RI_batch, phys_props, pad)
    return loss

def train_model(model, phys_props, optimizer,
                train_dataset, val_dataset,
                RI_monitor, epochs=50, pad=4):
    """
    Полный цикл обучения и валидации.
    В ходе тренировки каждые 10 эпох визуализируется поле на срезе.
    Возвращает словарь history с loss по эпохам.
    """
    history = {'train': [], 'val': []}
    start_time = time.time()

    for epoch in range(1, epochs+1):
        # --- Training loop ---
        train_losses = []
        for RI_batch in train_dataset:
            RI_batch = tf.cast(RI_batch, dtype=tf.float32)
            loss = train_step(model, RI_batch, phys_props, optimizer, pad)
            train_losses.append(loss.numpy())
        train_loss = np.mean(train_losses)

        # --- Validation loop ---
        val_losses = []
        for RI_batch in val_dataset:
            RI_batch = tf.cast(RI_batch, dtype=tf.float32)
            loss = val_step(model, RI_batch, phys_props, pad)
            val_losses.append(loss.numpy())
        val_loss = np.mean(val_losses)

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # Logging
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.3e}  val_loss={val_loss:.3e}  time={elapsed:.1f}s")
            # Inference and visualization on monitored RI
            generate_and_visualize(model, RI_monitor, phys_props, dx, phys_props.k0, pad)
            # Plot loss curves
            plt.figure(figsize=(6,4))
            plt.semilogy(history['train'], label='Train')
            plt.semilogy(history['val'],   label='Val')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.1f}s")
    return history

# Запуск обучения
history = train_model(model, phys_props, optimizer,
                      dataset_train, dataset_val,
                      RI_monitor, epochs=50, pad=4)

# 5) Final Evaluation и 2D-срез
print(f"\nFinal train loss: {history['train'][-1]:.2e}, final val loss: {history['val'][-1]:.2e}")

U_pred = model(RI_monitor, training=False).numpy()[0]
U_real = U_pred[..., 0]
U_imag = U_pred[..., 1]

mid_y = Ny // 2
slice_Re = U_real[mid_y, :, :]
slice_Im = U_imag[mid_y, :, :]

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
axs[0].imshow(slice_Re.T, origin='lower', cmap='inferno')
axs[0].set_title('Re(U_s) | Y-slice')
axs[0].axis('off')
axs[1].imshow(slice_Im.T, origin='lower', cmap='inferno')
axs[1].set_title('Im(U_s) | Y-slice')
axs[1].axis('off')
plt.tight_layout()
plt.show()
