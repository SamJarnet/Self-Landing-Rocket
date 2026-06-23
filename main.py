from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 
import numpy as np
from matplotlib.patches import Circle, Polygon


class Boids:
    def __init__(self):

        with plt.ioff():
            self.fig = plt.figure()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(10, 10)

        with plt.ioff():
            self.ax = plt.axes([0.1, 0.25, 0.8, 0.70], xlim=(0, 10), ylim=(0, 10))         # type: ignore

        vertices = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        triangle = Polygon(vertices.tolist(), fc='black', zorder=3)
        

        self.rocket = {"shape": triangle,"pos": np.array([5.0, 1.0]), "vel": np.array([0.0, 0.0])}  

        self.ax.axhline(y=1)
        self.dt = 0.1
        self.thrusting = False
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.key_release)


    def key_press(self, event):
        if event.key == 'w':
            self.thrusting = True

    def key_release(self, event):
        if event.key == 'w':
            self.thrusting = False

    def init(self):
        self.ax.add_patch(self.rocket["shape"])
        return []

    

    def step(self):
        if self.thrusting:
            self.rocket["vel"] += np.array([0.0, 0.02])

        self.rocket["vel"] += np.array([0.0, -0.01])
        self.rocket["pos"] += self.rocket["vel"] * self.dt

        if self.rocket["pos"][1] < 1.0:
            if abs(self.rocket["vel"][1]) > 1:
                print("boom")
            self.rocket["pos"][1] = 1.0
            self.rocket["vel"] = np.array([0.0, 0.0])
            
    
    def get_triangle_vertices(self, pos, vel, size):
        angle = np.arctan2(vel[1], vel[0])
        
        vertices = np.array([[size, 0.0],[-size, -size / 1.5],[-size, size / 1.5]])
        
        cos, sin = np.cos(angle), np.sin(angle)
        rotation= np.array([[cos, -sin],[sin, cos]]) # rotation matrix
        
        new_vertices = np.dot(vertices, rotation.T) + pos # rotate the vertices + shift pos
        return new_vertices

    def animate(self, j):

        self.step()
        pos = self.rocket["pos"]
        vel = self.rocket["vel"]
            
        new_vertices = self.get_triangle_vertices(pos, vel, 0.15)
        self.rocket["shape"].set_xy(new_vertices)
    

        return []
    
        

    def run(self):
        self.anim = FuncAnimation(self.fig, self.animate, init_func=self.init, frames=360, interval=20, blit=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    sim = Boids()
    sim.run()