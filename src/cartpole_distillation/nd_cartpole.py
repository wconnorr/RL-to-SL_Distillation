"""
An extension of OpenAI Gym's implentation of the cart-pole environment, vectorized into N dimensions. Their code (which forms the base of this code) can be found here: https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
"""

"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gymnasium as gym
from gymnasium import logger, spaces
from gymnasium.envs.classic_control import utils


class NDCartPoleEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):


    """
    Initializes the Cart Pole problem with n degrees of freedom. Note that standard Cart Pole has 1 degree of freedom.
    """
    def __init__(self, degrees_of_freedom=1):
        if not isinstance(degrees_of_freedom, int) or degrees_of_freedom < 1:
            raise Exception("Degrees of freedom must be an integer >= 1.")
        self.degrees_of_freedom = degrees_of_freedom
        
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                [
                    self.x_threshold * 2,
                    np.finfo(np.float32).max,
                    self.theta_threshold_radians * 2,
                    np.finfo(np.float32).max,
                ] for _ in range(degrees_of_freedom)
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2*self.degrees_of_freedom)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        """
        If we assume full acceleration in either direction and the cart comes to a stop after 1 step, which seems to be the case here:
        We only need to move the cart in 1 dimension per timestep. Cart acceleration=0 in all directions other than the action-dimension, which will be +/-force_mag, based on direction.
        The pole is different, it will continue accelerating due to gravity, but it will only accelerate due to the cart in one of the dimensions.
        State is a deg_freedom x 4 matrix
        """
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        terminated = False

        next_state = np.empty((self.degrees_of_freedom,4),dtype=np.float32)

        for i, dof_state in enumerate(self.state):

            x, x_dot, theta, theta_dot = dof_state
            if action//2 != i:
                force = 0
            elif action & 1 == 1:
                force = self.force_mag
            else:
                force = -self.force_mag
            costheta = math.cos(theta)
            sintheta = math.sin(theta)

            # For the interested reader:
            # https://coneural.org/florian/papers/05_cart_pole.pdf
            temp = (
                force + self.polemass_length * theta_dot**2 * sintheta
            ) / self.total_mass # f = ma :. a = f/m
            thetaacc = (self.gravity * sintheta - costheta * temp) / (
                self.length * (4.0 / 3.0 - self.masspole * costheta**2 / self.total_mass)
            ) # pole's angular acceleration
            xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass # cart's acceleration: if this is 0, velocity will stay the same.

    #       Cartpole always uses euler kinematics integrator
            x = x + self.tau * x_dot # pos_t = pos_{t-1} + 1timestep*velocity_{t-1}
            x_dot = x_dot + self.tau * xacc  # velocity_t = velocity_{t-1} + 1timestep*acc_t
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc

            next_state[i] = (x, x_dot, theta, theta_dot)

            terminated = terminated or bool(
                x < -self.x_threshold
                or x > self.x_threshold
                or theta < -self.theta_threshold_radians
                or theta > self.theta_threshold_radians
            )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0
        self.state = next_state
        return self.state, reward, terminated, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        # Note that if you use custom reset bounds, it may lead to out-of-bound
        # state/observations.
        low, high = utils.maybe_parse_reset_bounds(
            options, -0.05, 0.05  # default low
        )  # default high
        self.state = self.np_random.uniform(low=low, high=high, size=(self.degrees_of_freedom,4))
        self.steps_beyond_terminated = None

        return np.array(self.state, dtype=np.float32), {}

    def close(self):
        return # Nothing needs to be done to close the env
