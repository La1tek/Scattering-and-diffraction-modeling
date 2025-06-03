import numpy as np
import tensorflow as tf
import math

# ------------------------
# 2) Physical properties and operators
# ------------------------

def create_computational_grid(Ny, Nx, Nz, dy, dx, dz):
    # Возвращает массивы numpy X, Y, Z, определяющие вычислительную область.
    x = np.linspace(-Nx//2, Nx//2-1, Nx) * dx
    y = np.linspace(-Ny//2, Ny//2-1, Ny) * dy
    z = np.linspace(-Nz//2, Nz//2-1, Nz) * dz
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='xy')
    return XX, YY, ZZ

class PhysicalProperties:
    def __init__(self, Ny, Nx, Nz, wavelength, dy, dx, dz, n0):
        self.Ny = Ny
        self.Nx = Nx
        self.Nz = Nz
        self.wl = wavelength
        self.dy = dy
        self.dx = dx
        self.dz = dz
        self.n0 = n0
        self.k0 = 2 * math.pi / wavelength
        self.k = self.k0 * n0

    def phase_modulation(self):
        # e^{i k0 n0 z} падающая волна
        X, Y, Z = create_computational_grid(self.Ny, self.Nx, self.Nz, self.dy, self.dx, self.dz)
        phase = np.exp(1j * self.k * Z)
        phase = phase.reshape((1, self.Ny, self.Nx, self.Nz, 1))
        return tf.cast(phase, tf.complex64)

    def pml_coeffs(self, sigma=0.4, thickness=16, p=4):
        # PML - Perfectly Matched Layer
        X, Y, Z = create_computational_grid(self.Ny, self.Nx, self.Nz, self.dy, self.dx, self.dz)

        Lx = self.Nx * self.dx
        Ly = self.Ny * self.dy
        Lz = self.Nz * self.dz
        d = thickness * self.dx

        def s_factor(coord, L):
            # Returns complex stretching factor s = 1/(1 + i*sigma*((|c|-(L/2-d))^p))
            mask_in = np.abs(coord) < (L/2 - d)
            sigma_arr = 1 + 1j * sigma * ((np.abs(coord)-(L/2-d))**p)
            s = np.where(mask_in, 1.0, sigma_arr)
            return s

        sx = s_factor(X, Lx)
        sy = s_factor(Y, Ly)
        sz = s_factor(Z, Lz)

        # reshape to 5D tensors
        sx = sx.reshape((1, self.Ny, self.Nx, self.Nz, 1))
        sy = sy.reshape((1, self.Ny, self.Nx, self.Nz, 1))
        sz = sz.reshape((1, self.Ny, self.Nx, self.Nz, 1))

        return (tf.cast(1/sx, tf.complex64),
                tf.cast(1/sy, tf.complex64),
                tf.cast(1/sz, tf.complex64))


# ------------------------
# 3) Physics-informed loss operations
# ------------------------

def complex_mul(a, b):
    real = tf.math.real(a)*tf.math.real(b) - tf.math.imag(a)*tf.math.imag(b)
    imag = tf.math.real(a)*tf.math.imag(b) + tf.math.imag(a)*tf.math.real(b)
    return tf.complex(real, imag)

def finite_difference_kernels(order=5):
    # 4th-order finite-difference stencil (Fathy et al.)
    de = np.array([0.0, +1/24, -9/8, +9/8, -1/24], dtype=np.float32)
    dh = np.array([+1/24, -9/8, +9/8, -1/24, 0.0], dtype=np.float32)

    # build 3D kernels
    kshape = (5,5,5,1,1)
    ker = np.zeros(kshape, dtype=np.float32)
    center = 2

    # dx kernel along x
    kr = ker.copy(); kr[center,:,center,0,0] = de
    kh = ker.copy(); kh[center,:,center,0,0] = dh
    kx_e = tf.convert_to_tensor(kr)
    kx_h = tf.convert_to_tensor(kh)

    # dy kernel along y
    kr = ker.copy(); kr[:,center,center,0,0] = de
    kh = ker.copy(); kh[:,center,center,0,0] = dh
    ky_e = tf.convert_to_tensor(kr)
    ky_h = tf.convert_to_tensor(kh)

    # dz kernel along z
    kr = ker.copy(); kr[center,center,:,0,0] = de
    kh = ker.copy(); kh[center,center,:,0,0] = dh
    kz_e = tf.convert_to_tensor(kr)
    kz_h = tf.convert_to_tensor(kh)

    return kx_e, ky_e, kz_e, kx_h, ky_h, kz_h

def helmholtz_pinn_loss(U_total, RI_input, phys: PhysicalProperties, pad=4):
    """
    Полный PINN-loss по уравнению Гельмгольца с учётом:
      - трёх компонент вторых производных,
      - ε·U,
      - term ~ (n^2−n0^2)·UI,
      - PML-маскирование,
      - усреднённое по всей области.
    """

    # 1) Собираем комплексное поле
    U = tf.complex(U_total[...,0], U_total[...,1])              # shape=(1,Ny,Nx,Nz)
    UI = phys.phase_modulation()                                # carrier e^{ik0 n0 z}
    Uenv = U * tf.math.conj(tf.squeeze(UI, axis=-1))
                                # медленно меняющаяся огибающая

    # 2) Готовим ядра и PML
    kx_e, ky_e, kz_e, kx_h, ky_h, kz_h = finite_difference_kernels()
    sx, sy, sz = phys.pml_coeffs()

    # 3) Вспомогательная функция для одной оси
    def second_derivative_along(field, ke, kh, s):
        field5d = tf.expand_dims(field, axis=-1)           # (B, Ny, Nx, Nz, 1)
        # первая свертка (edge):
        r1 = tf.nn.conv3d(field5d, ke, strides=(1,1,1,1,1), padding='SAME')
        i1 = tf.nn.conv3d(field5d, ke, strides=(1, 1, 1, 1, 1), padding='SAME')
        E1 = complex_mul(s, tf.complex(r1, i1))
        # вторая свертка (h):
        r2 = tf.nn.conv3d(tf.math.real(E1), kh, strides=(1, 1, 1, 1, 1), padding='SAME')
        i2 = tf.nn.conv3d(tf.math.imag(E1), kh, strides=(1, 1, 1, 1, 1), padding='SAME')
        return complex_mul(s, tf.complex(r2, i2)) / phys.dx**2

    # 4) Считаем Del² Uenv по трём осям
    Lx = second_derivative_along(Uenv, kx_e, kx_h, sx)
    Ly = second_derivative_along(Uenv, ky_e, ky_h, sy)
    Lz = second_derivative_along(Uenv, kz_e, kz_h, sz)
    Lap = Lx + Ly + Lz

    # 5) Правый член (n^2 − n0^2)·UI
    n2 = (RI_input + phys.n0)**2
    rhs = -phys.k0**2 * (n2 - phys.n0**2) * UI

    # 6) Полный residual
    R = Lap + rhs

    # 7) PML-маскирование: обнуляем границы толщиной pad
    mask = np.ones((1, phys.Ny, phys.Nx, phys.Nz, 1), np.float32)
    mask[:,:,:,:pad,:] = 0
    mask[:,:,:,-pad:,:] = 0
    R = R * tf.cast(mask, tf.complex64)

    # 8) Финальный loss
    return tf.reduce_mean(tf.abs(R)**2)
