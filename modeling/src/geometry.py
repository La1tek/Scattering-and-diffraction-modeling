import meep as mp

class Aperture:
    """
    Базовый класс для апертур.
    Параметры:
      extent: [xmin, xmax, ymin, ymax]
      d: толщина по X
      aperture_distance: расстояние от xmin до апертуры по X
    """
    def __init__(self, extent, d, aperture_distance, epsilon=9999):
        self.xmin, self.xmax, self.ymin, self.ymax = extent
        self.d = d
        self.ap_dist = aperture_distance
        self.epsilon = epsilon
        self._compute_center()

    def _compute_center(self):
        # Вычисляем центр области
        cx = (self.xmax + self.xmin) / 2
        cy = (self.ymax + self.ymin) / 2
        self.center = mp.Vector3(cx, cy)
        # Позиция апертуры по X в системе Meep
        self.x_pos = self.ap_dist - cx

    def geometry(self):
        """
        Должен возвращать список mp.GeometricObject
        """
        raise NotImplementedError("geometry() must be implemented by subclasses")


class SphereAperture(Aperture):
    def __init__(self, extent, aperture_radius, aperture_distance, epsilon=9999):
        super().__init__(extent, d=mp.inf, aperture_distance=aperture_distance, epsilon=epsilon)
        self.radius = aperture_radius

    def geometry(self):
        return [
            mp.Cylinder(
                radius=self.radius,
                height=mp.inf,
                center=mp.Vector3(self.x_pos, 0, 0),
                material=mp.Medium(epsilon=self.epsilon)
            )
        ]


class SingleSlitAperture(Aperture):
    def __init__(self, extent, b, d, aperture_distance, epsilon=9999):
        super().__init__(extent, d, aperture_distance, epsilon)
        self.b = b  # ширина щели по Y

    def geometry(self):
        total_y = self.ymax - self.ymin
        half_slit = self.b / 2
        # Верхняя часть
        top = mp.Block(
            size=mp.Vector3(self.d, total_y/2 - half_slit, mp.inf),
            center=mp.Vector3(self.x_pos, (half_slit + total_y/2)/2, 0),
            material=mp.Medium(epsilon=self.epsilon)
        )
        # Нижняя часть
        bottom = mp.Block(
            size=mp.Vector3(self.d, total_y/2 - half_slit, mp.inf),
            center=mp.Vector3(self.x_pos, -(half_slit + total_y/2)/2, 0),
            material=mp.Medium(epsilon=self.epsilon)
        )
        return [top, bottom]


class DoubleSlitAperture(Aperture):
    def __init__(self, extent, b, s, d, aperture_distance, epsilon=9999):
        super().__init__(extent, d, aperture_distance, epsilon)
        self.b = b  # ширина каждой щели
        self.s = s  # расстояние между центрами щелей

    def geometry(self):
        total_y = self.ymax - self.ymin
        half_slit = self.b / 2
        half_sep = self.s / 2
        # Верхняя часть
        top = mp.Block(
            size=mp.Vector3(self.d, total_y/2 - (half_sep + half_slit), mp.inf),
            center=mp.Vector3(self.x_pos, (half_sep + half_slit + total_y/2)/2, 0),
            material=mp.Medium(epsilon=self.epsilon)
        )
        # Средняя часть (между щелями)
        middle = mp.Block(
            size=mp.Vector3(self.d, self.s - self.b, mp.inf),
            center=mp.Vector3(self.x_pos, 0, 0),
            material=mp.Medium(epsilon=self.epsilon)
        )
        # Нижняя часть
        bottom = mp.Block(
            size=mp.Vector3(self.d, total_y/2 - (half_sep + half_slit), mp.inf),
            center=mp.Vector3(self.x_pos, -(half_sep + half_slit + total_y/2)/2, 0),
            material=mp.Medium(epsilon=self.epsilon)
        )
        return [top, middle, bottom]


class EllipseAperture(Aperture):
    def __init__(self, extent, rx, ry, d, aperture_distance, epsilon=9999):
        super().__init__(extent, d, aperture_distance, epsilon)
        self.rx = rx
        self.ry = ry

    def geometry(self):
        return [
            mp.Ellipsoid(
                size=mp.Vector3(2*self.rx, 2*self.ry, self.d),
                center=mp.Vector3(self.x_pos, 0, 0),
                material=mp.Medium(epsilon=self.epsilon)
            )
        ]


class TriangularAperture(Aperture):
    def __init__(self, extent, base, height, d, aperture_distance, epsilon=9999):
        super().__init__(extent, d, aperture_distance, epsilon)
        self.base = base
        self.height = height

    def geometry(self):
        v1 = mp.Vector3(0, -self.base/2, 0)
        v2 = mp.Vector3(0, +self.base/2, 0)
        v3 = mp.Vector3(0,       0, self.height)
        return [
            mp.Prism(
                [v1, v2, v3],
                self.d,
                mp.Vector3(1, 0, 0),
                mp.Vector3(self.x_pos, 0, 0),
                mp.Medium(epsilon=self.epsilon)
            )
        ]


class GratingAperture(Aperture):
    def __init__(self, extent, pitch, width, height, d, aperture_distance, epsilon=9999):
        super().__init__(extent, d, aperture_distance, epsilon)
        self.pitch = pitch
        self.width = width
        self.height = height

    def geometry(self):
        blocks = []
        count = int((self.ymax - self.ymin) / self.pitch) + 1
        for i in range(-count, count+1):
            y = i * self.pitch
            blocks.append(
                mp.Block(
                    size=mp.Vector3(self.d, self.width, self.height),
                    center=mp.Vector3(self.x_pos, y, 0),
                    material=mp.Medium(epsilon=self.epsilon)
                )
            )
        return blocks