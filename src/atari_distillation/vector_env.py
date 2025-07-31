""""
Produces vectorized environments for both ND Cart-pole and Atari, allowing n parallel environments to be used in reinforcement learning.
"""
import gymnasium as gym
import numpy as np

from CustomVectorEnv import SyncVectorEnv


# ale_py keeps track of Atari ROMs
import ale_py
# must register ale_py envs to access Atari ROMs
gym.register_envs(ale_py)

def make_atari_vector_env(num_envs, envname):
  global atariname
  atariname = envname
  return gym.vector.SyncVectorEnv([
      _make_atari
      for i in range(num_envs)])

def _make_atari():
  env = gym.wrappers.AtariPreprocessing(gym.make(atariname), scale_obs=True)
  env = gym.wrappers.RecordEpisodeStatistics(gym.wrappers.FrameStackObservation(env, 4))
  return env

def make_atari(atariname):
  env = gym.wrappers.AtariPreprocessing(gym.make(atariname), scale_obs=True)
  env = gym.wrappers.RecordEpisodeStatistics(gym.wrappers.FrameStackObservation(env, 4))
  return env