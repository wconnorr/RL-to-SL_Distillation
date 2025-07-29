""""
Produces vectorized environments for both ND Cart-pole and Atari, allowing n parallel environments to be used in reinforcement learning.
"""

import gym

def make_atari_vector_env(num_envs, envname):
  return gym.vector.SyncVectorEnv([
      _make_atari(envname)
      for i in range(num_envs)])

def _make_atari(envname):
  def curry():
    env = gym.wrappers.AtariPreprocessing(gym.make(envname), scale_obs=True)
    env = gym.wrappers.RecordEpisodeStatistics(gym.wrappers.FrameStack(env, 4))
    return env
  return curry

# Terminates the environment when the total reward reaches r
class RewardTerminationWrapper(gym.Wrapper):
  def __init__(self, env, r):
    super().__init__(env)
    self.env = env
    self.reward_cutoff = r
    self.running_reward = 0
  def reset(self):
    self.running_reward = 0
    return self.env.reset()
  def step(self, action):
    next_state, reward, done, t, info = self.env.step(action)
    self.running_reward += reward
    if self.running_reward >= self.reward_cutoff:
      done = True
    return next_state, reward, done, t, info
