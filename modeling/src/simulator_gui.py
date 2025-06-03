import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import os
import meep as mp
from apertures import *
from geometry import (
    SphereAperture,
    SingleSlitAperture,
    DoubleSlitAperture,
    EllipseAperture,
    TriangularAperture,
    GratingAperture,
)
from visualization import Visualizer
import sys

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

class ApertureSimulatorApp(tk.Tk):
    # Геометрическое отображение параметров для каждого типа апертуры
    geometric_mapping = {
        'Sphere':      ['radius','ap_dist','epsilon'],
        'Single_Slit': ['slit_width','thickness','ap_dist','epsilon'],
        'Double_Slit': ['slit_width','separation','thickness','ap_dist','epsilon'],
        'Ellipse':     ['rx','ry','thickness','ap_dist','epsilon'],
        'Grating':     ['pitch','grating_width','height','thickness','ap_dist','epsilon'],
    }

    def __init__(self):
        super().__init__()
        self.title("Scattering Simulator")
        self.create_widgets()

    def create_widgets(self):
        container = ttk.Frame(self, padding=10)
        container.grid(row=0, column=0, sticky='nsew')

        # --- Geometry Inputs ---
        geom_frame = ttk.LabelFrame(container, text="Dimensions", padding=5)
        geom_frame.grid(row=0, column=0, columnspan=4, sticky='ew', pady=5)
        ttk.Label(geom_frame, text="xmin:").grid(row=0, column=0, sticky='w')
        self.xmin = tk.DoubleVar(value=0.0)
        ttk.Entry(geom_frame, textvariable=self.xmin, width=10).grid(row=0, column=1)
        ttk.Label(geom_frame, text="xmax:").grid(row=0, column=2, sticky='w')
        self.xmax = tk.DoubleVar(value=60.0)
        ttk.Entry(geom_frame, textvariable=self.xmax, width=10).grid(row=0, column=3)
        ttk.Label(geom_frame, text="ymin:").grid(row=1, column=0, sticky='w')
        self.ymin = tk.DoubleVar(value=-15.0)
        ttk.Entry(geom_frame, textvariable=self.ymin, width=10).grid(row=1, column=1)
        ttk.Label(geom_frame, text="ymax:").grid(row=1, column=2, sticky='w')
        self.ymax = tk.DoubleVar(value=15.0)
        ttk.Entry(geom_frame, textvariable=self.ymax, width=10).grid(row=1, column=3)

        ap_frame = ttk.LabelFrame(container, text="Aperture", padding=5)
        ap_frame.grid(row=1, column=0, columnspan=4, sticky='ew', pady=5)
        ttk.Label(ap_frame, text="Aperture Type:").grid(row=2, column=0, sticky='w')
        self.ap_type = tk.StringVar(value="Sphere")
        ap_cb = ttk.Combobox(ap_frame, textvariable=self.ap_type, state='readonly',
                             values=list(self.geometric_mapping.keys()))
        ap_cb.grid(row=2, column=1, columnspan=3, sticky='ew')

        self.param_frame = ttk.Frame(ap_frame)
        self.param_frame.grid(row=3, column=0, columnspan=4, pady=5, sticky='nsew')
        self.vars = {k: tk.DoubleVar(value=v) for k,v in {
            'radius':2.0,'slit_width':1.0,'thickness':0.5,'separation':3.0,
            'rx':3.0,'ry':5.0,'base':2.0,'height':3.0,'pitch':2.0,
            'grating_width':1.0,'ap_dist':30.0,'epsilon':9999.0
        }.items()}
        labels = {
            'radius':'Radius:','slit_width':'Slit Width:','thickness':'Thickness:',
            'separation':'Separation:','rx':'Radius X:','ry':'Radius Y:',
            'base':'Base:','height':'Height:','pitch':'Pitch:',
            'grating_width':'Grating Width:','ap_dist':'Aperture Distance:','epsilon':'Epsilon:'
        }
        self.param_widgets = {}
        for k,t in labels.items():
            lbl = ttk.Label(self.param_frame, text=t)
            ent = ttk.Entry(self.param_frame, textvariable=self.vars[k])
            self.param_widgets[k] = (lbl, ent)

        # --- Source Inputs ---
        src_frame = ttk.LabelFrame(container, text="Source Inputs", padding=5)
        src_frame.grid(row=2, column=0, columnspan=4, sticky='ew', pady=5)
        ttk.Label(src_frame, text="Source Type:").grid(row=0, column=0, sticky='w')
        self.src_type = tk.StringVar(value="Gaussian")
        src_cb = ttk.Combobox(src_frame, textvariable=self.src_type, state='readonly',
                              values=["Gaussian","Incoherent Rectangular"])
        src_cb.grid(row=0, column=1, columnspan=3, sticky='ew')

        self.src_frame = ttk.Frame(src_frame)
        self.src_frame.grid(row=1, column=0, columnspan=4, pady=5, sticky='nsew')
        self.src_vars = {
            'wavelength':tk.DoubleVar(value=0.65),
            'beam_width':tk.DoubleVar(value=30.0),
            'bandwidth':tk.DoubleVar(value=0.65),
            'dipoles':tk.IntVar(value=100)
        }
        src_labels = {
            'wavelength':'Wavelength [nm]:','beam_width':'Beam Width:',
            'bandwidth':'Bandwidth:','dipoles':'Number of Dipoles:'
        }
        self.src_widgets = {}
        for k,t in src_labels.items():
            lbl = ttk.Label(self.src_frame, text=t)
            ent = ttk.Entry(self.src_frame, textvariable=self.src_vars[k])
            self.src_widgets[k] = (lbl, ent)

        # --- Simulation Parameters ---
        sim_frame = ttk.LabelFrame(container, text="Simulation Parameters", padding=5)
        sim_frame.grid(row=3, column=0, columnspan=4, sticky='ew', pady=5)
        ttk.Label(sim_frame, text="PPW:").grid(row=0, column=0, sticky='w')
        self.ppw = tk.IntVar(value=10)
        ttk.Entry(sim_frame, textvariable=self.ppw, width=10).grid(row=0, column=1)
        ttk.Label(sim_frame, text="Total fs:").grid(row=0, column=2, sticky='w')
        self.total_fs = tk.DoubleVar(value=300.0)
        ttk.Entry(sim_frame, textvariable=self.total_fs, width=10).grid(row=0, column=3)
        ttk.Label(sim_frame, text="Frames:").grid(row=1, column=0, sticky='w')
        self.nframes = tk.IntVar(value=50)
        ttk.Entry(sim_frame, textvariable=self.nframes, width=10).grid(row=1, column=1)
        ttk.Label(sim_frame, text="GIF FPS:").grid(row=1, column=2, sticky='w')
        self.fps = tk.IntVar(value=5)
        ttk.Entry(sim_frame, textvariable=self.fps, width=10).grid(row=1, column=3)

        # --- Output Options ---
        flags_frame = ttk.LabelFrame(container, text="Output Options", padding=5)
        flags_frame.grid(row=4, column=0, columnspan=4, sticky='ew', pady=5)
        self.flag_field = tk.BooleanVar(value=True)
        self.flag_gif = tk.BooleanVar(value=True)
        opts_out = [
            ('Field Slider', self.flag_field),
            ('Save GIF', self.flag_gif),
        ]
        self.cb_widgets = {}
        for i,(txt,var) in enumerate(opts_out):
            cb = ttk.Checkbutton(flags_frame, text=txt, variable=var)
            cb.grid(row=i, column=0, sticky='w')
            self.cb_widgets[txt] = cb
            
        # --- Graph Options ---
        graph_frame = ttk.LabelFrame(container, text="Graphs Options", padding=5)
        graph_frame.grid(row=5, column=0, columnspan=4, sticky='ew', pady=5)
        self.flag_lineI = tk.BooleanVar(value=True)
        self.flag_lineEz = tk.BooleanVar(value=False)
        self.flag_scatter = tk.BooleanVar(value=False)
        self.flag_circular = tk.BooleanVar(value=False)
        opts = [
            ('Line I', self.flag_lineI),
            ('Line Re(Ez)', self.flag_lineEz),
            ('Scattering Pattern', self.flag_scatter),
            ('Circular I', self.flag_circular)
        ]
        for i,(txt,var) in enumerate(opts):
            cb = ttk.Checkbutton(graph_frame, text=txt, variable=var)
            cb.grid(row=i, column=0, sticky='w')
            self.cb_widgets[txt] = cb

        # --- Actions ---
        btn_frame = ttk.Frame(container)
        btn_frame.grid(row=6, column=0, columnspan=4, pady=10)
        ttk.Button(btn_frame, text="Run Simulation", command=self.run_simulation).pack(side='left', padx=5)
        ttk.Button(btn_frame, text="Clear Log", command=self.clear_log).pack(side='left', padx=5)

        # --- Log ---
        log_frame = ttk.LabelFrame(container, text="Log Output", padding=5)
        log_frame.grid(row=7, column=0, columnspan=4, sticky='ew', pady=5)
        self.log = scrolledtext.ScrolledText(log_frame, width=40, height=10, state='disabled')
        self.log.grid(row=0, column=0)
        sys.stdout = TextRedirector(self.log)
        sys.stderr = TextRedirector(self.log)

        # привязка инициализации полей
        self.ap_type.trace_add('write', lambda *args: self.update_geometry_fields())
        self.src_type.trace_add('write', lambda *args: self.update_source_fields())
        self.update_source_fields()
        self.update_geometry_fields()

    def update_geometry_fields(self):
        ap = self.ap_type.get()
        # скрываем все
        for lbl,ent in self.param_widgets.values():
            lbl.grid_forget(); ent.grid_forget()
        # показываем нужные
        for i,key in enumerate(self.geometric_mapping.get(ap, [])):
            lbl,ent = self.param_widgets[key]
            lbl.grid(row=i, column=0, sticky='w')
            ent.grid(row=i, column=1)
        # корректируем состояния чекбоксов
        if ap in ['Single_Slit','Double_Slit','Grating']:
            self.cb_widgets['Circular I'].configure(state='normal')
            self.cb_widgets['Scattering Pattern'].configure(state='disabled')
            self.flag_scatter.set(False)
        else:
            self.cb_widgets['Scattering Pattern'].configure(state='normal')
            self.cb_widgets['Circular I'].configure(state='disabled')
            self.flag_circular.set(False)

    def update_source_fields(self):
        st = self.src_type.get()
        for lbl,ent in self.src_widgets.values():
            lbl.grid_forget(); ent.grid_forget()
        keys = ['wavelength','beam_width'] if st=='Gaussian' else ['wavelength','bandwidth','dipoles']
        for i,key in enumerate(keys):
            lbl,ent = self.src_widgets[key]
            lbl.grid(row=i, column=0, sticky='w')
            ent.grid(row=i, column=1)

    def clear_log(self):
        self.log.config(state='normal')
        self.log.delete('1.0', tk.END)
        self.log.yview_moveto(0)
        self.log.config(state='disabled')

    def run_simulation(self):
        try:
            self.clear_log()
            self.log.config(state='normal')
            self.log.insert(tk.END, "Running simulation...\n")
            self.log.config(state='disabled')

            ap_key = self.ap_type.get()
            ext = [self.xmin.get(), self.xmax.get(), self.ymin.get(), self.ymax.get()]
            params = {k: self.vars[k].get() for k in self.vars}

            # создаём объект апертуры
            aprts = {
                'Sphere':SphereAperture,
                'Single_Slit':SingleSlitAperture,
                'Double_Slit':DoubleSlitAperture,
                'Ellipse':EllipseAperture,
                'Grating':GratingAperture
            }
            spec = {
                'Sphere':{'aperture_radius':params['radius']},
                'Single_Slit':{'b':params['slit_width'],'d':params['thickness']},
                'Double_Slit':{'b':params['slit_width'],'s':params['separation'],'d':params['thickness']},
                'Ellipse':{'rx':params['rx'],'ry':params['ry'],'d':params['thickness']},
                'Grating':{'pitch':params['pitch'],'width':params['grating_width'],'height':params['height'],'d':params['thickness']}
            }[ap_key]
            aperture = aprts[ap_key](
                extent=ext,
                aperture_distance=params['ap_dist'],
                epsilon=params['epsilon'],
                **spec
            )
            geom = aperture.geometry()

            # создаём источник
            if self.src_type.get() == 'Gaussian':
                src = GaussianSource(
                    position=mp.Vector3(ext[0]+1,0,0),
                    direction=mp.Vector3(1,0,0),
                    λ=self.src_vars['wavelength'].get(),
                    beam_width=self.src_vars['beam_width'].get()
                )
            else:
                src = Incoherent_rectangular_source(
                    number_of_dipoles=self.src_vars['dipoles'].get(),
                    position=mp.Vector3(ext[0]+1,0,0),
                    λ=self.src_vars['wavelength'].get(),
                    bandwidth=self.src_vars['bandwidth'].get(),
                    width=self.src_vars['bandwidth'].get(),
                    height=ext[3]-ext[2]
                )

            # строим строку параметров только из нужных ключей
            keys = self.geometric_mapping[ap_key]
            param_str = '_'.join(f"{k}-{params[k]}" for k in sorted(params) if k in keys)

            folder = f"{ap_key}_{param_str}"
            os.makedirs(folder, exist_ok=True)
            h5_path = os.path.join(folder, f"{folder}.h5")

            # запускаем симуляцию
            aperture_simulation(
                light_source=src,
                extent=ext,
                geometry=geom,
                pixels_per_wavelength=self.ppw.get(),
                total_femtoseconds=self.total_fs.get(),
                number_of_frames=self.nframes.get(),
                simulation_name=h5_path
            )

            vis = Visualizer(h5_path)
            if self.flag_field.get():     vis.plot_field_slider()
            if self.flag_gif.get():       vis.save_gif(save_path=os.path.join(folder,'vis.gif'), fps=self.fps.get())
            if self.flag_lineI.get():     vis.plot_line_I(save_path=os.path.join(folder,'line_I(y).png'))
            if self.flag_lineEz.get():    vis.plot_line_Ez(save_path=os.path.join(folder,'line_Ez(y).png'))
            if self.flag_scatter.get():
                params = {k: self.vars[k].get() for k in self.vars}
                if self.ap_type.get() == 'Sphere':
                    vis.plot_scattering_pattern(radius = params['radius']+0.01, save_path=os.path.join(folder,'scatter.png'))
                else:
                    vis.plot_scattering_pattern(radius = max(params['rx'], params['ry'])+0.01, save_path=os.path.join(folder,'scatter.png'))
            if self.flag_circular.get():  vis.plot_circular_I(save_path=os.path.join(folder,'circular.png'))

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    app = ApertureSimulatorApp()
    app.mainloop()
