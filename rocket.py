from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation 
import numpy as np
from matplotlib.patches import Circle, Polygon
from matplotlib.widgets import Button
from bandit import Bandit

class Rocket:
    def __init__(self):

        with plt.ioff():
            self.fig = plt.figure()
        self.fig.set_dpi(100)
        self.fig.set_size_inches(10, 20)

        with plt.ioff():
            self.ax = plt.axes([0.1, 0.25, 0.8, 0.70], xlim=(0, 10), ylim=(0, 20))         # type: ignore

        vertices = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        triangle = Polygon(vertices.tolist(), fc='black', zorder=3)
        

        self.rocket = {"shape": triangle,"pos": np.array([2.0, 1.0]), "vel": np.array([0.0, 0.0]), "angle" : 0.0, "angular_vel":0.0}  


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

        # bandit-controlled landing mode
        self.bandit = None
        self.bandit_Q = None
        self.bandit_landing = False
        self.bandit_train_episodes = 15000

        self.setup_widgets() 
    
    def setup_widgets(self): 
        land = plt.axes([0.15, 0.15, 0.1, 0.04]) # type: ignore 
        self.land_button = Button(land, 'land', hovercolor='0.775') 
        self.land_button.on_clicked(self.set_land)

        bandit_land = plt.axes([0.30, 0.15, 0.18, 0.04]) # type: ignore
        self.bandit_land_button = Button(bandit_land, 'bandit land', hovercolor='0.775')
        self.bandit_land_button.on_clicked(self.set_bandit_land)

    def key_press(self, event):
        if event.key == 'w':
            self.thrusting = True
        elif event.key == 'd':
            self.rocket["angular_vel"] -= 0.01
        elif event.key == 'a':
            self.rocket["angular_vel"] += 0.01

    def key_release(self, event):
        if event.key == 'w':
            self.thrusting = False


    def set_land(self, event):
        self.bandit_landing = False
        self.landing = True

    def set_bandit_land(self, event):
        self.landing = False

        if self.bandit_Q is None:
            print("training bandit...")
            self.train_bandit()
            print("training complete")

        self.bandit_landing = True


    def init(self):
        self.ax.add_patch(self.rocket["shape"])

        return []


    def train_bandit(self):
        self.bandit = Bandit(self)
        self.bandit_Q = self.bandit.learn(self.bandit_train_episodes)

        self.rocket["pos"] = np.array([2.0, 1.0])
        self.rocket["vel"] = np.array([0.0, 0.0])
        self.rocket["angle"] = 0.0
        self.rocket["angular_vel"] = 0.0

    def bandit_action(self):
        state = self.bandit.get_state()
        if state not in self.bandit_Q:
            return 0
        return int(np.argmax(self.bandit_Q[state]))

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
    
    def thrust(self, angle):
        thrust_x = -np.sin(angle) * self.thrust_speed
        thrust_y = np.cos(angle) * self.thrust_speed
        self.rocket["vel"] += np.array([thrust_x, thrust_y])

    def step(self):
        if self.bandit_landing:
            action = self.bandit_action()
            _, _, done = self.bandit.step(action)
            if done:
                self.bandit_landing = False
            return

        self.rocket["angle"] += self.rocket["angular_vel"]
        self.rocket["angular_vel"] *= 0.98
        if self.thrusting:
            self.thrust(self.rocket["angle"])

        if self.landing:
            x_error = self.landing_pad[0] - self.rocket["pos"][0]
            goal_x = x_error * 0.01 - self.rocket["vel"][0] * 0.1

            k = 5
            goal_angle = -goal_x * k

            # limit angle 
            goal_angle = np.clip(goal_angle, -0.4, 0.4)

            angle_error = goal_angle - self.rocket["angle"]
            self.rocket["angular_vel"] += (0.02 * angle_error- 0.2 * self.rocket["angular_vel"])

        if self.land() and self.landing:
            self.thrust(self.rocket["angle"])


        # update gravity and position
        self.rocket["vel"] += np.array([0.0, -0.01])
        self.rocket["pos"] += self.rocket["vel"] * self.dt

        if self.rocket["pos"][1] < 1.0:
            if abs(self.rocket["vel"][1]) > 0.5:
                print("boom")
            self.landing_burn, self.landing = False, False
            self.rocket["pos"][1] = 1.0
            self.rocket["vel"] = np.array([0.0, 0.0])

        


    def rotate(self, pos, size, angle):
        vertices = np.array([[0.0, size], [-size / 1.5, -size], [size / 1.5, -size]])

        cos, sin = np.cos(angle), np.sin(angle)
        rotation = np.array([[cos, -sin], [sin, cos]])
        
        new_vertices = np.dot(vertices, rotation.T) + pos
        return new_vertices

    def animate(self, j):
        self.step()
        
        new_vertices = self.rotate(self.rocket["pos"], 0.15, self.rocket["angle"])
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