import gymnasium as gym

from vec_normalize import VecNormalize

def _make_mujoco(env_name):
  def curry():
    env = gym.make(env_name)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env
  return curry

def make_mujoco_vector_env(num_envs, env_name):
  return VecNormalize(
    gym.vector.SyncVectorEnv([_make_mujoco(env_name)
                              for i in range(num_envs)]))

