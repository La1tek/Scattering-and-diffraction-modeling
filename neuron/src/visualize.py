import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from physics import helmholtz_pinn_loss, PhysicalProperties

# ------------------------
# 6) Inference и визуализация
# ------------------------
def generate_and_visualize(model, RI_input, phys_props, dx, k0, pad=4):
    """
    Выполняет предсказание поля по входному RI, вычисляет PINN-loss и
    визуализирует центральный Y-срез комплексного поля.

    Аргументы:
      model: tf.keras.Model, MaxwellNet3D
      RI_input: np.ndarray или tf.Tensor формы (1, Ny, Nx, Nz, 1)
      phys_props: экземпляр класса PhysicalProperties
      dx, k0: шаг сетки (dx) и волновое число (k0)
      pad: толщина PML для обнуления граничных значений
    """

    RI_tensor = tf.convert_to_tensor(RI_input, dtype=tf.float32)  # (1,Ny,Nx,Nz,1)

    U_pred = model(RI_tensor, training=False)  # форма (1,Ny,Nx,Nz,2)

    loss_val = helmholtz_pinn_loss(U_pred, RI_tensor, phys_props, pad)

    U_np = U_pred.numpy()
    Ny   = U_np.shape[1]

    slice_complex = U_np[0, Ny//2, :, :, 0] + 1j * U_np[0, Ny//2, :, :, 1]

    # Визуализация
    plt.figure(figsize=(5,5))
    plt.title(f"PINN Loss = {loss_val.numpy():.2e}")
    plt.imshow(np.real(slice_complex).T, origin='lower', cmap='inferno')
    plt.axis('off')
    plt.colorbar(label='Re(Us)')
    plt.show()
