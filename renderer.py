import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

class RocketRenderer:
    def __init__(self, render_mode="human", landing_pad=(6.0, 1.0)):
        self.render_mode = render_mode
        self.landing_pad = np.array(landing_pad, dtype=np.float64)

        self.fig = None
        self.ax = None
        self.rocket_patch = None

    def _init_fig(self):
        if self.render_mode == "human":
            plt.ion()

        self.fig, self.ax = plt.subplots(figsize=(5, 10), dpi=80) # Lower DPI/size
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 20)

        self.ax.axhline(y=1, color='black')
        self.ax.scatter(self.landing_pad[0], self.landing_pad[1], color='red', s=100)

        vertices = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        self.rocket_patch = Polygon(vertices, fc='black', zorder=3)
        self.ax.add_patch(self.rocket_patch)

    @staticmethod
    def rotate(pos, size, angle):
        vertices = np.array([[0.0, size], [-size / 1.5, -size], [size / 1.5, -size]])
        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin], [sin, cos]])
        return np.dot(vertices, rotation.T) + pos

    def render(self, pos, angle):
        if self.fig is None:
            self._init_fig()

        new_vertices = self.rotate(pos, 0.15, angle)
        self.rocket_patch.set_xy(new_vertices)

        if self.render_mode == "human":
            plt.pause(0.001) 

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
            self.rocket_patch = None