import random
import numpy as np
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from rocket import Rocket
from collections import defaultdict

class Bandit:
    def __init__(self, env: "Rocket"):
        self.env = env
        self.grid_shape = np.array([10.0, 20.0]) # now 5 by 5 continuous instead of graph
        self.start = np.array([0.0, 0.0])
        self.goal = np.array([6.0, 1.0]) # 10 by 5 with bins
        self.actions = {
            0: "nothing",
            1: "thrust",
            2: "rotate_left",
            3: "rotate_right"
        }



    def get_state(self):
        x = np.clip(self.env.rocket["pos"][0], 0, self.grid_shape[0] - 0.1)
        y = np.clip(self.env.rocket["pos"][1], 0, self.grid_shape[1] - 0.1)
        
        # clip so bins don't break
        vel_x = np.clip(self.env.rocket["vel"][0], -5.0, 5.0)
        vel_y = np.clip(self.env.rocket["vel"][1], -5.0, 5.0)
        angle = np.clip(self.env.rocket["angle"], -np.pi, np.pi)
        
        x_bin = int(x / 0.2)
        y_bin = int(y / 0.5)
        vel_x_bin = int((vel_x + 5) / 0.2)
        vel_y_bin = int((vel_y + 5) / 0.2)
        angle_bin = int((angle + np.pi) / 0.1)
        
        return (x_bin, y_bin, vel_x_bin, vel_y_bin, angle_bin)
    
    def reset(self):
        self.env.rocket["pos"] = self.start.copy()
        self.env.rocket["vel"] = np.array([0.0, 0.0])
        self.env.rocket["angle"] = 0.0
        self.env.rocket["angular_vel"] = 0.0

    
    def do_action(self, action):
        to_do = self.actions[action]
        if to_do == "thrust":
            self.env.thrust(self.env.rocket["angle"])
        elif to_do == "rotate_right":
            self.env.rocket["angular_vel"] -= 0.01
        elif to_do == "rotate_left":
            self.env.rocket["angular_vel"] += 0.01
        

    def step(self, action):
        reward = -1
        done = False

        self.env.rocket["angle"] += self.env.rocket["angular_vel"]
        self.env.rocket["angular_vel"] *= 0.98

        self.do_action(action)

        self.env.rocket["vel"][1] -= self.env.gravity
        future = self.env.rocket["pos"] + self.env.rocket["vel"] * self.env.dt

        if 0.0 < future[0] < self.grid_shape[0] and 0.0 < future[1] < self.grid_shape[1]:
            self.env.rocket["pos"] = future
            vel = self.env.rocket["vel"]
            if np.linalg.norm(self.env.rocket["pos"] - self.goal) < 0.05 and vel[0]< 0.05 and vel[1]< 0.05:
                reward = 100
                done = True
            else:
                reward = -100
                done = True

        state = self.get_state()
        return state, reward, done


    def get_action(self, state, Q, epsilon):
        if random.random() < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[state])
        return action

    def learn(self, eps):
        Q = defaultdict(lambda: np.zeros(4))
        alpha, gamma = 0.1, 0.95
        epsilon = 0.05

        for i in range(0, eps):
            self.reset()
            done = False
            state = self.get_state()

            while not done:

                action = self.get_action(state, Q, epsilon)

                next_state, reward, done = self.step(action)

                Q[state][action] += alpha * (reward+ gamma * max(Q[next_state])- Q[state][action])
                state = next_state
            epsilon = max(0.01, epsilon * 0.995)
        return Q
