import gym

from nd_cartpole import NDCartPoleEnv

def _make_cartpole():
  def curry():
    env = gym.make("CartPole-v1")
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
  return curry

def make_cartpole_vector_env(num_envs):
  return gym.vector.SyncVectorEnv([
      _make_cartpole()
      for i in range(num_envs)])

def make_atari_vector_env(num_envs, envname, simplify=False):
  return gym.vector.SyncVectorEnv([
      _make_atari(envname, simplify)
      for i in range(num_envs)])

def _make_ndcartpole(degrees_of_freedom=1, end_reward=None):
  def curry():
    env = NDCartPoleEnv(degrees_of_freedom)
    if end_reward is not None:
      env = RewardTerminationWrapper(env, end_reward)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
  return curry

def make_ndcartpole_vector_env(num_envs, degrees_of_freedom=1, end_reward=None):
  return gym.vector.SyncVectorEnv([
      _make_ndcartpole(degrees_of_freedom, end_reward)
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
