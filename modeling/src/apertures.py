import meep as mp
import h5py
import os
from pathlib import Path
import numpy as np
from lights import *

def aperture_simulation(light_source,
                               extent,
                               geometry,
                               pixels_per_wavelength,
                               total_femtoseconds,
                               number_of_frames,
                               simulation_name):


    femtoseconds_per_frame = total_femtoseconds / number_of_frames

    # центр ячейки и её размеры
    center = mp.Vector3((extent[1] + extent[0]) / 2,
                        (extent[2] + extent[3]) / 2)
    cell   = mp.Vector3(extent[1] - extent[0],
                        extent[3] - extent[2],
                        0)

    # PML-слои
    pml_layers = [mp.PML(1)]

    # источники
    sources = light_source.get_meep_sources(center)

    pixels_per_um = pixels_per_wavelength / light_source.λ

    sim = mp.Simulation(cell_size=cell,
                        sources=sources,
                        boundary_layers=pml_layers,
                        geometry=geometry,
                        resolution=pixels_per_um,
                        force_complex_fields=False)

    # удаляем старый файл, если есть
    if os.path.exists(Path(simulation_name)):
        os.remove(Path(simulation_name))

    with h5py.File(simulation_name, 'a') as f:
        # первый кадр
        sim.run(until=femtoseconds_per_frame * 3. / 10)
        ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
        eps = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Dielectric)
        sx  = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Sx)

        # создаём датасеты
        f.create_dataset('Ez',      (number_of_frames, *ez.shape),   dtype='float64', maxshape=(None, *ez.shape))
        f.create_dataset('eps',     eps.shape,                      dtype='float64')
        f.create_dataset('Sx',      (number_of_frames, *sx.shape),   dtype='float64', maxshape=(None, *sx.shape))

        # атрибуты
        f.attrs['cell_x']             = cell.x
        f.attrs['cell_y']             = cell.y
        f.attrs['extent']             = np.array(extent)
        f.attrs['total_femtoseconds'] = total_femtoseconds
        f.attrs['number_of_frames']   = number_of_frames
        f.attrs['simulated_femtoseconds'] = femtoseconds_per_frame
        f.attrs['simulated_frames']   = 1

        # заполняем первый кадр
        f['eps'][:]   = eps
        absEz        = np.abs(ez)
        f['Ez'][0]   = absEz
        f['Sx'][0]   = sx
        # сохраняем max как атрибут, а не dataset
        f.attrs['max_Ez_value'] = absEz.max()

        # последующие кадры
        for i in range(1, number_of_frames):
            sim.run(until=femtoseconds_per_frame * 3. / 10)
            ez = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Ez)
            sx = sim.get_array(center=mp.Vector3(), size=cell, component=mp.Sx)
            f['Ez'][i] = np.abs(ez)
            f['Sx'][i] = sx

            current_max = np.abs(ez).max()
            # обновляем атрибут, если необходимо
            if f.attrs['max_Ez_value'] < current_max:
                f.attrs['max_Ez_value'] = current_max

            f.attrs['simulated_femtoseconds'] += femtoseconds_per_frame
            f.attrs['simulated_frames']       += 1
        
        # поле на линии Y = 10.0
        x_target = 60
        x_local = x_target - center.x

        Ez_line_x = sim.get_array(
            center=mp.Vector3(x_local, 0, 0),
            size=mp.Vector3(0, cell.y, 0),
            component=mp.Ez
        )

        f.create_dataset('Ez_Y_10', data=Ez_line_x)
        f.attrs['Ez_Y_10_y_coord'] = x_target
