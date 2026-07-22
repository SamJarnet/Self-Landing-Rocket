import gymnasium as gym
from gymnasium import spaces
import numpy as np

from physics import RocketPhysics
from renderer import RocketRenderer

class RocketEnv(gym.Env):

    def __init__(self, render_mode=None):
        super().__init__()
        self.physics = RocketPhysics()
        self.renderer = RocketRenderer(render_mode=render_mode) if render_mode else None

        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),
            dtype=np.float32
        )
        self.steps = 0
        
    def get_observation(self):
        return np.array([
            self.physics.pos[0] / 10.0,
            self.physics.pos[1] / 20.0,
            self.physics.vel[0] / 5.0,
            self.physics.vel[1] / 5.0,
            self.physics.angle / np.pi,
            self.physics.angular_vel / 2.0
        ], dtype=np.float32)
    
    def get_reward(self, action):
        pos_x, pos_y = self.physics.pos
        vel_x, vel_y = self.physics.vel
        angle = self.physics.angle

        target_x, target_y = self.physics.landing_pad

        x_error = abs(pos_x - target_x)
        y_error = abs(pos_y - target_y)

        reward = -0.15 * x_error - 0.1 * y_error - 0.1 * abs(angle)

        if pos_y < 5.0 and vel_y < 0:
            reward -= 1.0 * (vel_y ** 2)

        if action in [1, 4, 5]:
            reward -= 0.05

        if self.physics.landed:
            landing_error = abs(pos_x - target_x)
            accuracy_bonus = max(0.0, 300.0 - (100.0 * landing_error))
            reward += accuracy_bonus

        if self.physics.crashed:
            reward -= 150.0

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize starting altitude closer to the pad during training
        start_y = np.random.uniform(9.0, 12.0)
        start_x = np.random.uniform(3.0, 7.0)
        
        self.physics.reset(pos=(start_x, start_y))
        self.steps = 0
        return self.get_observation(), {}
    
    def step(self, action):
        self.steps += 1
        crashed, landed = self.physics.step(action)

        observation = self.get_observation()
        reward = self.get_reward(action)
        terminated = crashed or landed
        truncated = self.steps >= 400

        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.renderer:
            self.renderer.render(self.physics.pos, self.physics.angle)

    def close(self):
        if self.renderer is not None:
            self.renderer.close()