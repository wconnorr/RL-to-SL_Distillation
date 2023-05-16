"""
Validation using a fully-trained actor.
"""

import os
import time
import argparse

import numpy as np

import torch
import torch.nn.functional as F

from torch.distributions.categorical import Categorical

import gym

from models.cartpole_models import Actor
from envs.nd_cartpole import NDCartPoleEnv
from envs.vector_env import RewardTerminationWrapper

# Cartpole

# ACTION SPACE: steer cart [left, right]
ACTION_SPACE = 2
# STATE SPACE:  [x_cart, vx_cart, theta_pole_, vtheta_pole] (not necessarily in that order)
STATE_SPACE  = 4

device = None

# Return GPU if available, else return CPU
# If assert_gpu, quit the program if the GPU is not available
def select_device(assert_gpu=False):
  if assert_gpu:
    assert torch.cuda.is_available()
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
  begin = time.time()
    
  # Parser arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("-t", "--num_trials", help="number of RL trails to run for each learner after training: required for randomness in action selection and in environment behavior", type=int, default=1)
  parser.add_argument("--load_from", help="load models from provided folder: will look for policy_sd.pt")
  parser.add_argument("--degrees_of_freedom", help="degrees of freedom for nd cartpole environment. Standard (2d) cart pole has 1 degree of freedom. Value must be an integer > 0.", type=int, default=1)
  parser.add_argument("result_dir", help="path to save results plots")
  args = parser.parse_args()
  print("EXPERIMENT: ", args)
    
  global device, STATE_SPACE, ACTION_SPACE
    
  if args.device:
    device = torch.device(args.device)
  else:
    device = select_device()
    print("Device not selected, using device:", device)
    
  results_path = args.result_dir

  ## RL MODES ##
#   randomize_inner_architecture = args.randomize_inner_architecture
  
  ## HYPERPARAMETERS ##
  
  reward_threshold = 500 # max running reward

  if args.degrees_of_freedom <= 1:
    env = gym.make("CartPole-v1") 
  else:
    env = RewardTerminationWrapper(NDCartPoleEnv(args.degrees_of_freedom), reward_threshold)
    STATE_SPACE  = 4*args.degrees_of_freedom
    ACTION_SPACE = 2*args.degrees_of_freedom

  num_trials = args.num_trials
 
  rewards = [None] * num_trials
   
  # Reinitialize inner network
  actor = Actor(state_size=STATE_SPACE, action_size=ACTION_SPACE).to(device)
 
  load(actor, os.path.join(args.load_from, "policy_sd.pt"))   
  
  for trial in range(num_trials):
    state, _ = env.reset()
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    done = False
    cumulative_reward = 0
    while not done:
      action = act(actor, state)
      state, reward, done, _, _ = env.step(action.item())
      state = torch.from_numpy(state).float().unsqueeze(0).to(device)
      cumulative_reward += reward
    rewards[trial] = cumulative_reward

  # TODO
  # The only output we really need is a list (over the learners) of lists (over the trials) of rewards
  # What about removing randomness (at least in ACT)?
  save(results_path, rewards, args)

def save(path, rewards, args):
    if not os.path.exists(path):
      os.makedirs(path)
    with open(os.path.join(path, "out.txt"), 'w') as f:
      f.write("EXPERIMENT: " + str(args) + '\n\n')
      f.write(str(rewards) + "\tAVG:" + str(np.mean(rewards)) + "\tMAX:" + str(max(rewards)) + "\n")


def load(model, file_path):
    """
    model must be initialized to the correct type/size!
    file_path must point to the model's state dict .pt file!
    
    The state dict is saved on the device used to train the model that run. map_location loads it into the currently-used training device!
    """
    model.load_state_dict(torch.load(file_path, map_location=device))

# Policy Gradient Methods

def act(actor, state):
  with torch.no_grad():
    policy = actor(state)
    probs = Categorical(logits=policy)
    action = probs.sample()
  return action#, probs.log_prob(action), probs.entropy(), value

def calculate_losses(policy_network, value_network, transition, epsilon, encoder=None):
  states, actions, prior_policy, _, _, _, returns, advantages, entropies = transition

  if encoder is None:
    current_policy = policy_network(states)[F.one_hot(actions.long(), ACTION_SPACE).bool()]
  else:
    current_policy = policy_network(encoder(states))[F.one_hot(actions.long(), ACTION_SPACE).bool()]
  current_values = value_network(states)

  # calculate ratio quicker this way, rather than softmaxing them both
  ratio = (current_policy - prior_policy).exp()

  # normalize advantages
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-epsilon, 1+epsilon)).mean()
  # TODO: Try clipping value loss and see if it works
  value_loss = F.mse_loss(current_values.squeeze(), returns)
    
  # TODO: Add entropy loss back in

  return policy_loss, value_loss, entropies.mean()

# Performs 1 rollout of fixed length of the agent acting in the environment.
def perform_rollout(agent, critic, vec_env, rollout, rollout_len, state, encoder=None):
  with torch.no_grad():
    final_rewards = []
    # TODO: Remove values if we don't need v^0 for advantage calculations
    states, actions, prior_policy, rewards, dones, values, _, _, entropies = rollout

    _, num_envs, state_shape = states.shape

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, action_distribution, entropy, value = act(agent, critic, state.to(device), encoder=encoder)

      # Env takes step based on action
      next_state, reward, done, _, info = vec_env.step(action.cpu().numpy())

      # Store step for learning
      states[i] = state.view(num_envs, state_shape)
      actions[i] = action
      prior_policy[i] = action_distribution
      rewards[i] = torch.from_numpy(reward)
      dones[i] = torch.from_numpy(done)
      values[i] = value.squeeze(1)
      entropies[i] = entropy
    
      state = torch.from_numpy(next_state)
      
      if isinstance(info, dict) and 'final_info' in info.keys():
        epis = [a for a in info['final_info'] if a is not None]
        for item in epis:
          final_rewards.append(item['episode']['r'])
      else:
        for item in info:
          if "episode" in item.keys():
            final_rewards.append(item['episode']['r']) 
    
  return final_rewards, state, done # no need to return rollout, its updated in-place

# Calculates advantage and return, bootstrapping using value when environment has not terminated
# Modifies them in-place in the rollout
def general_advantage_estimation(critic, rollout, next_state, next_done, gamma, gae_lambda):
  _, _, _, rewards, dones, values, returns, advantages, _ = rollout
  rollout_len = rewards.size(0)
  # NEED: values, next_state, next_done?, 
  #   values = value(state) at each iteration of the rollout
  # next_state = state at the end of the rollout that hasn't been placed in states
  # next_done = same as next_state

  with torch.no_grad():
    next_value = critic(next_state.to(device)).reshape(1,-1)
#     advantages = torch.zeros_like(rewards).to(device)
    last_lambda = 0
    
    nextnonterminal = 1. - torch.from_numpy(next_done).float().to(device)
    nextvalues = next_value
    delta = rewards[rollout_len-1] + gamma * nextvalues*nextnonterminal - values[rollout_len-1]
    advantages[rollout_len-1] = last_lambda = delta # + gamma * gae_lambda * nextnonterminal * last_lambda # last_lambda = 0, so this is 0
    for t in reversed(range(rollout_len-1)):
      nextnonterminal = 1.0 - dones[t+1]
      nextvalues = values[t+1]
      delta = rewards[t] + gamma * nextvalues*nextnonterminal - values[t]
      advantages[t] = last_lambda = delta + gamma * gae_lambda * nextnonterminal * last_lambda
    returns = advantages + values
    rollout[6] = returns
    
if __name__ == '__main__':
  main()
