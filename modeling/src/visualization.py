import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import animation
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
import os

class Visualizer:
    """
    Класс для визуализации результатов FDTD-симуляции Meep из HDF5-файла.

    Методы:
      plot_field_slider(save_path=None, ...) - интерактивный просмотр Ez слайдером и сохранение скриншота
      save_gif(save_path, fps, ...)         - сохранение анимации поля Ez в GIF
      plot_line_Ez(save_path=None, ...)     - построение линейного среза Re(Ez) и сохранение
      plot_line_I(save_path=None, ...)      - построение линейного среза интенсивности I и сохранение
      plot_scattering_pattern(save_path)    - полярный график рассеяния вдоль окружности и сохранение
      plot_circular_I(save_path=None, ...)  - кольцевой (кольца Ньютона) график интенсивности и сохранение
    """
    def __init__(self, simulation_name):
        self.simulation_name = simulation_name
        # Загрузка всех необходимых данных один раз
        with h5py.File(simulation_name, 'r') as f:
            self.extent = f.attrs['extent']
            self.total_fs = f.attrs['total_femtoseconds']
            self.n_frames = f.attrs['number_of_frames']
            self.max_Ez = f.attrs['max_Ez_value']
            self.cell_y = f.attrs['cell_y']
            # Загружаем массивы
            self.Ez = f['Ez'][:]        # shape = (frames, nx, ny)
            self.eps = f['eps'][:]      # диэлектрический профиль

    def plot_field_slider(self, save_path: str | None = None, max_colormap_factor=1.0, colormap='twilight_shifted'):
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10,5))
        plt.subplots_adjust(bottom=0.2, left=0.15)
        ax.imshow(self.eps.T, interpolation='spline36', cmap='bone', alpha=1, extent=self.extent)
        im = ax.imshow(np.abs(self.Ez[0].T),
                       interpolation='spline36', cmap=colormap, alpha=0.9,
                       extent=self.extent,
                       vmax=self.max_Ez * max_colormap_factor)
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')

        def _update(val):
            t = slider.val
            idx = int(round(t / self.total_fs * (self.n_frames - 1)))
            im.set_data(np.abs(self.Ez[idx].T))
            fig.canvas.draw_idle()

        ax_slider = plt.axes([0.1, 0.05, 0.8, 0.05])
        slider = Slider(ax_slider, 't [fs]', 0, self.total_fs, valinit=0, color='#5c05ff')
        slider.on_changed(_update)

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Field slider snapshot saved: {save_path}")

    def save_gif(self, save_path: str, fps=10, max_colormap_factor=1.0, colormap='twilight_shifted'):
        fig, ax = plt.subplots(figsize=(10,5))
        ax.imshow(self.eps.T, interpolation='spline36', cmap='bone', alpha=1, extent=self.extent)
        im = ax.imshow(np.abs(self.Ez[0].T),
                       interpolation='spline36', cmap=colormap, alpha=0.9,
                       extent=self.extent,
                       vmax=self.max_Ez * max_colormap_factor)
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')

        def _animate(i):
            im.set_data(np.abs(self.Ez[i].T))
            return [im]

        anim = animation.FuncAnimation(fig, _animate,
                                       frames=self.n_frames,
                                       interval=1000/fps,
                                       blit=True)
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)

        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)


        anim.save(save_path, writer='pillow', fps=fps)
        plt.close(fig)
        print(f"GIF saved: {save_path}")

    def plot_line_Ez(self, dataset='Ez_Y_10', save_path: str | None = None):
        with h5py.File(self.simulation_name, 'r') as f:
            line = f[dataset][:]
            y_coord = f.attrs[f'{dataset}_y_coord']
        y = np.linspace(-self.cell_y/2, self.cell_y/2, len(line))
        fig, ax = plt.subplots(figsize=(5,10))
        ax.plot(np.real(line), y)
        ax.set_title(f'Re(Ez) along Y at X = {y_coord} μm')
        ax.set_xlabel('Re(Ez) [V/m]')
        ax.set_ylabel('Y [μm]')
        ax.grid(True)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Line plot Re(Ez) saved: {save_path}")
        else:
            default = f"{dataset}_ReEz.png"
            fig.savefig(default, dpi=300)
            print(f"Line plot Re(Ez) saved: {default}")
        plt.close(fig)

    def plot_line_I(self, dataset='Ez_Y_10', save_path: str | None = None):
        with h5py.File(self.simulation_name, 'r') as f:
            line = f[dataset][:]
            y_coord = f.attrs[f'{dataset}_y_coord']
        y = np.linspace(-self.cell_y/2, self.cell_y/2, len(line))
        fig, ax = plt.subplots(figsize=(5,10))
        ax.plot(np.real(line)**2, y)
        ax.invert_xaxis()
        ax.set_title(f'I along Y at X = {y_coord} μm')
        ax.set_xlabel('I [V²/m²]')
        ax.set_ylabel('Y [μm]')
        ax.grid(True)
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Line plot I saved: {save_path}")
        else:
            default = f"{dataset}_I.png"
            fig.savefig(default, dpi=300)
            print(f"Line plot I saved: {default}")
        plt.close(fig)

    def plot_scattering_pattern(
        self,
        radius: float = 2,
        center: tuple[float, float] = (30.0, 0.0),
        frame: int = -1,
        resolution: int = 360 * 3,
        smoothing_sigma: float = 2.0,
        save_path: str = None
    ):
        """
        Вычисляет и отображает рассеяние вдоль окружности радиусом `radius`.
        Returns theta и I_theta.
        """
        if frame < 0:
            frame = self.n_frames - 1

        Ez = self.Ez[frame]
        nx, ny = Ez.shape

        x = np.linspace(self.extent[0], self.extent[1], nx)
        y = np.linspace(self.extent[2], self.extent[3], ny)

        interpolator = RegularGridInterpolator((x, y), Ez, method='linear', bounds_error=False, fill_value=0)

        theta = np.linspace(0, 2 * np.pi, resolution, endpoint=False)
        xs = center[0] + radius * np.cos(theta)
        ys = center[1] + radius * np.sin(theta)
        points = np.column_stack((xs, ys))

        Ez_vals = interpolator(points)
        I_theta = np.abs(Ez_vals) ** 2
        if smoothing_sigma > 0:
            I_theta = gaussian_filter1d(I_theta, sigma=smoothing_sigma, mode='wrap')

        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax.plot(theta, I_theta)
        ax.set_title(f"Scattering intensity at r = {radius:.2f} μm", va='bottom')
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Scattering pattern saved: {save_path}")
        plt.show()
        plt.close(fig)
        return theta, I_theta

    def plot_circular_I(self, dataset='Ez_Y_10', resolution=500, save_path: str | None = None):
        """
        Строит кольцевой график интенсивности.
        """
        # Читаем одномерный профиль вдоль Y
        with h5py.File(self.simulation_name, 'r') as f:
            line = f[dataset][:]
            y_coord = f.attrs[f'{dataset}_y_coord']
        # Интенсивность вдоль линии
        I_line = np.real(line)**2

        # Получаем координаты Y и оставляем только неотрицательные радиусы
        y = np.linspace(-self.cell_y/2, self.cell_y/2, len(I_line))
        mask = y >= 0
        r_vals = y[mask]
        I_r = I_line[mask]

        # Интерполятор I(r)
        from scipy.interpolate import interp1d
        interp = interp1d(r_vals, I_r, bounds_error=False, fill_value=0.0)

        # Двумерная сетка (X, Y) и радиальная координата R
        x = np.linspace(-self.cell_y/2, self.cell_y/2, resolution)
        X, Y = np.meshgrid(x, x)
        R = np.sqrt(X**2 + Y**2)

        # Применяем интерполяцию ко всему массиву R
        I2D = interp(R)

        # Рисуем
        fig, ax = plt.subplots(figsize=(6,6))
        im = ax.imshow(
            I2D,
            origin='lower',
            extent=[-self.cell_y/2, self.cell_y/2, -self.cell_y/2, self.cell_y/2],
            interpolation='bilinear',
            cmap='inferno'
        )
        ax.set_title(f'Circular intensity from {dataset} at X = {y_coord} μm')
        ax.set_xlabel('X [μm]')
        ax.set_ylabel('Y [μm]')
        cbar = fig.colorbar(im, label='I [V²/m²]')
        plt.tight_layout()

        # Сохранение
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=300)
            print(f"Circular plot saved: {save_path}")
        else:
            default = f"{dataset}_circular.png"
            fig.savefig(default, dpi=300)
            print(f"Circular plot saved: {default}")

        plt.close(fig)
