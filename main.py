from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 
import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button

class Rocket:
    def __init__(self):

        with plt.ioff():
            self.fig = plt.figure()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(10, 10)

        with plt.ioff():
            self.ax = plt.axes([0.1, 0.25, 0.8, 0.70], xlim=(0, 10), ylim=(0, 30))         # type: ignore

        vertices = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        triangle = Polygon(vertices.tolist(), fc='black', zorder=3)
        

        self.rocket = {"shape": triangle,"pos": np.array([2.0, 1.0]), "vel": np.array([0.0, 0.0])}  


        self.landing_pad = np.array([6, 1])
        self.ax.axhline(y=1)
        self.ax.scatter(self.landing_pad[0], self.landing_pad[1], color='red', s=100)


        self.dt = 0.1
        self.thrust_speed = 0.02
        self.gravity = 0.01

        self.thrusting = False
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        self.fig.canvas.mpl_connect('key_release_event', self.key_release)
        self.landing = False
        self.landing_burn = False
        self.setup_widgets() 
    
    def setup_widgets(self): 
        land = plt.axes([0.15, 0.15, 0.1, 0.04]) # type: ignore 
        self.land_button = Button(land, 'land', hovercolor='0.775') 
        self.land_button.on_clicked(self.set_land)

    def key_press(self, event):
        if event.key == 'w':
            self.thrusting = True

    def key_release(self, event):
        if event.key == 'w':
            self.thrusting = False


    def set_land(self, event):
        self.landing = True


    def init(self):
        self.ax.add_patch(self.rocket["shape"])

        return []


    def land(self):
        height = self.rocket["pos"][1] - 1.0

        if self.rocket["vel"][1] >= 0:
            return False

        vel = abs(self.rocket["vel"][1])
        acc = (self.thrust_speed - self.gravity)
        
        stopping_distance = (vel ** 2) / (2 * acc)
        return height <= stopping_distance


    def lookahead(self):
        future_pos = self.rocket["pos"].copy()
        future_vel = self.rocket["vel"].copy()
        crash = False
        crash_vel = np.array([0.0, 0.0])
        crash_pos = np.array([0.0, 0.0])

        crash_step = 0
        for i in range(1000):
            future_vel += np.array([0.0, -self.gravity])
            future_pos += future_vel * self.dt

            if future_pos[1] < 1.0:
                if abs(future_vel[1]) > 0.5:
                    crash = True
                    crash_vel = future_vel
                    crash_step = i
                    crash_pos = future_pos
                future_pos[1] = 1.0
                future_vel = np.array([0.0, 0.0])

        return crash_pos

    def step(self):
        if self.thrusting:
            self.rocket["vel"] += np.array([0.001, self.thrust_speed])

        if self.landing and not self.landing_burn:
            self.landing_burn = self.land()
        
        elif self.landing_burn:
            if self.rocket["vel"][1] >= 0:
                self.landing_burn = False
        
        if self.landing_burn:
            print(self.lookahead()) 

            x = self.rocket["pos"][0]
            y = self.rocket["pos"][1]
            vel_y = self.rocket["vel"][1]
            
            burn_time = (y - 1.0) / abs(vel_y) if abs(vel_y) > 0.001 else 1.0
            change = (self.landing_pad[0] - x) / burn_time

            x_change = (change - self.rocket["vel"][0]) * 0.1
            y_change = self.thrust_speed

            self.rocket["vel"] += np.array([x_change, y_change])

            angle = np.arctan2(y_change, x_change) - np.pi / 2
        else:
            mod_v = np.linalg.norm(self.rocket["vel"])
            if mod_v > 0.05: 
                angle = np.arctan2(self.rocket["vel"][1], self.rocket["vel"][0]) - np.pi / 2
            else:
                angle = 0.0

        # update gravity and position
        self.rocket["vel"] += np.array([0.0, -0.01])
        self.rocket["pos"] += self.rocket["vel"] * self.dt

        if self.rocket["pos"][1] < 1.0:
            if abs(self.rocket["vel"][1]) > 0.5:
                print("boom")
            self.landing_burn, self.landing = False, False
            self.rocket["pos"][1] = 1.0
            self.rocket["vel"] = np.array([0.0, 0.0])

        return angle


    def rotate(self, pos, size, angle):
        vertices = np.array([[0.0, size], [-size / 1.5, -size], [size / 1.5, -size]])

        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin], [sin, cos]])
        
        new_vertices = np.dot(vertices, rotation.T) + pos
        return new_vertices

    def animate(self, j):
        angle = self.step()
        new_vertices = self.rotate(self.rocket["pos"], 0.15, angle)
        self.rocket["shape"].set_xy(new_vertices)
        return []
    
        

    def run(self):
        self.anim = FuncAnimation(self.fig, self.animate, init_func=self.init, frames=360, interval=20, blit=False)
        try:
            plt.show()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    sim = Rocket()
    sim.run()