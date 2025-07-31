"""
Uses PPO to directly learn Atari (RL, no distillation)
"""

import os
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.distributions.categorical import Categorical
from itertools import chain

from atari_models import Actor, Critic, RLDataset
import vector_env

# Atari

envs = ['MsPacmanNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'CentipedeNoFrameskip-v4', 'PongNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'PongNoFrameskip-v4']
env_name = envs[4]

device = None

# Return GPU if available, else return CPU
# If assert_gpu, quit the program if the GPU is not available
def select_device(assert_gpu=False):
  if assert_gpu:
    assert torch.cuda.is_available()
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
  # Parser arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("-p", "--policy_epochs", help="number of epochs per learning cycle", type=int, default=4)
  parser.add_argument("--save_models", help="save models in results-dir", action="store_true")
  parser.add_argument("--environment", help="Environment to be used, defaults to 'BreakoutNoFrameskip-v4'", default=None)
  parser.add_argument("result_dir", help="path to save results plots")
  args = parser.parse_args()
  print("EXPERIMENT: ", args)
    
  global device, env_name
  if args.environment:
    env_name = args.environment
    
  if args.device:
    device = torch.device(args.device)
  else:
    device = select_device()
    print("Device not selected, using device:", device)
    
  results_path = args.result_dir


  env = vector_env.make_atari(env_name)

  global n_actions
  n_actions = env.action_space.n
  init_screen, _ = env.reset()
  c, h, w = init_screen.shape
  print(env_name)
  print(n_actions, "actions")
  print("Screen size (stacked and resized):",init_screen.shape)


  ## RL MODES ##
  ## HYPERPARAMETERS ##
  # OUTER HYPERPARAMETERS
  lr = 2.5e-4
#   num_epochs = 10000
  num_envs = 10 # number of parallel environments to gather data from
  rollout_len = 200 # number of steps to take in the envs per rollout
  policy_epochs = args.policy_epochs
  rl_batch_size = 512 # Batch size can work with a wide range of values
  gamma = .99
  gae_lambda = .95
  epsilon = .1
    
  max_grad_norm = .5

  entropy_coefficient = .01
  value_coefficient = .5
    
  adam_eps = 1e-5 # default parameter is 1e-8

  ## LISTS FOR PERFORMANCE ANALYSIS ##
  # OUTER STATISTICS
  end_rewards = []
  policy_losses = []
  value_losses = []
  entropy_losses = []

  env = vector_env.make_atari_vector_env(num_envs, env_name)
    
  policy_network = Actor(c, n_actions).to(device)
  value_network = Critic(c).to(device)
    
  if args.save_models:
    if not os.path.exists(os.path.join(results_path,"init")):
        os.makedirs(os.path.join(results_path,"init"))
    torch.save(policy_network.state_dict(), os.path.join(results_path, "init", 'policy_sd.pt'))
    torch.save(value_network.state_dict(), os.path.join(results_path, "init", 'value_sd.pt'))

  optimizer = optim.Adam(chain(policy_network.parameters(), value_network.parameters()), lr=lr, eps=adam_eps)


  # Rollout data structures: constant length based on number of steps and envs.
  # No need to define these each rollout session, just overwrite them!
  states = torch.zeros((rollout_len, num_envs, c, h, w)).to(device)
  actions = torch.zeros((rollout_len, num_envs)).to(device)
  prior_policy = torch.zeros((rollout_len, num_envs)).to(device) # stored for chosen action only!
  rewards = torch.zeros((rollout_len, num_envs)).to(device)
  dones = torch.zeros((rollout_len, num_envs)).to(device)
  values = torch.zeros((rollout_len, num_envs)).to(device)
  returns = None # this will be overwritten with a new tensor, rather than being filled piecewise
  advantages = torch.zeros((rollout_len, num_envs)).to(device)
  entropies = torch.zeros((rollout_len, num_envs)).to(device)
    
  rollout = [states, actions, prior_policy, rewards, dones, values, returns, advantages, entropies]

  step = 0
    
  last_state, _ = env.reset()
  last_state = torch.from_numpy(last_state)

  epoch = 0
    
  while(True):
    
    gather_data = True
    
    for policy_epoch in range(policy_epochs):
      epoch_done = False
      reset_iter = True
      memit = 0
      while not epoch_done:
        memit += 1

        ### PPO OUTER TRAINING ###
        # Gather data with first net only!
        if gather_data:
          memory = []
          epoch_rewards = []
          # TODO: find out how best to do rollout rewards
          rollout_rewards, last_state, last_done = perform_rollout(policy_network, value_network, env, rollout, rollout_len, last_state)
            
          general_advantage_estimation(value_network, rollout, last_state, last_done, gamma, gae_lambda)
        
          end_rewards.extend(rollout_rewards)
          memory_dataloader = DataLoader(RLDataset(rollout), batch_size=rl_batch_size, shuffle=True)
          gather_data = False
            
        if reset_iter:
          memory_iter = iter(memory_dataloader)
          transition = next(memory_iter)
          reset_iter = False
    
        # Calculate and accumulate losses
        policy_loss, value_loss, entropy_loss = calculate_losses(policy_network, value_network, transition, epsilon)
        loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_loss
        loss.backward()
        nn.utils.clip_grad_norm_(chain(policy_network.parameters(), value_network.parameters()), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())
        entropy_losses.append(entropy_loss.item())
        
        step += 1
    

        try:
          transition = next(memory_iter)
        except:
          epoch_done = True
    
#     if epoch % 100 == 99:
#       loop.set_description("Epoch: {} Last Reward: {}".format(epoch, int(end_rewards[-1])))
#       loop.update(100)
    if epoch % 1000 == 999:
      plot(os.path.join(results_path, str(epoch+1)), end_rewards, policy_losses, value_losses, entropy_losses)
      if args.save_models:
        torch.save(policy_network.state_dict(), os.path.join(results_path, str(epoch+1), 'policy_sd.pt'))
        torch.save(value_network.state_dict(), os.path.join(results_path, str(epoch+1), 'value_sd.pt'))
    epoch += 1
#   loop.close()

#   print("CLOSED AFTER", step, "STEPS")
    
    
def plot(path, rewards, policy_losses, value_losses, entropy_losses):
    if not os.path.exists(path):
      os.makedirs(path)
    
    print("PLOTTING TO " + str(path))
    
    fig = plt.figure()
    plt.plot(rewards)
    plt.plot([e for e in pd.Series.rolling(pd.Series(rewards), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(rewards), 100).mean()])
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Cumulative Rewards")
    fig.savefig(os.path.join(path, "reward.png"), dpi=fig.dpi)
    
    fig = plt.figure()
    plt.plot(policy_losses)
    plt.plot([e for e in pd.Series.rolling(pd.Series(policy_losses), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(policy_losses), 100).mean()])
    plt.xlabel("Optimization Step")
    plt.ylabel("PPO Policy Loss")
    plt.title("Policy Losses")
    fig.savefig(os.path.join(path, "policy_loss.png"), dpi=fig.dpi)

    fig = plt.figure()
    plt.plot(value_losses)
    plt.plot([e for e in pd.Series.rolling(pd.Series(value_losses), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(value_losses), 100).mean()])
    plt.xlabel("Optimization Step")
    plt.ylabel("PPO Value Loss")
    plt.title("Value Losses")
    fig.savefig(os.path.join(path, "value_loss.png"), dpi=fig.dpi)
    
    fig = plt.figure()
    plt.plot(entropy_losses)
    plt.plot([e for e in pd.Series.rolling(pd.Series(entropy_losses), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(entropy_losses), 100).mean()])
    plt.xlabel("Optimization Step")
    plt.ylabel("Policy Entropy Loss")
    plt.title("Entropy Losses")
    fig.savefig(os.path.join(path, "entropy_loss.png"), dpi=fig.dpi)

    plt.close('all')


# Policy Gradient Methods

def act(actor, critic, state):
  with torch.no_grad():
    value = critic(state)
    policy = actor(state)
    probs = Categorical(logits=policy)
    action = probs.sample()
  return action, probs.log_prob(action), probs.entropy(), value

def calculate_losses(policy_network, value_network, transition, epsilon):
  states, actions, prior_policy, _, _, _, returns, advantages, entropies = transition

  current_policy = policy_network(states)[F.one_hot(actions.long(), n_actions).bool()]
  current_values = value_network(states).squeeze(1)

  # calculate ratio quicker this way, rather than softmaxing them both
  ratio = (current_policy - prior_policy).exp()

  # normalize advantages
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-epsilon, 1+epsilon))
  policy_loss = policy_loss.mean()
  # TODO: Try clipping value loss and see if it works
  value_loss = F.mse_loss(current_values, returns)
    
  # TODO: Add entropy loss back in

  return policy_loss, value_loss, entropies.mean()

# Performs 1 rollout of fixed length of the agent acting in the environment.
def perform_rollout(agent, critic, vec_env, rollout, rollout_len, state):
  with torch.no_grad():
    final_rewards = []
    # TODO: Remove values if we don't need v^0 for advantage calculations
    states, actions, prior_policy, rewards, dones, values, _, _, entropies = rollout

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, action_distribution, entropy, value = act(agent, critic, state.to(device))

      # Env takes step based on action
      next_state, reward, term, trunc, info = vec_env.step(action.cpu().numpy())

      done = np.logical_or(term, trunc)
        
      # Store step for learning
      states[i] = state
      actions[i] = action
      prior_policy[i] = action_distribution
      rewards[i] = torch.from_numpy(reward)
      dones[i] = torch.from_numpy(done)
      values[i] = value.squeeze(1)
      entropies[i] = entropy
    
      state = torch.from_numpy(next_state)
    
      if isinstance(info, dict) and 'episode' in info.keys():
        for r, finished in zip(info['episode']['r'], info['_episode']):
          if finished:
            final_rewards.append(int(r))
    
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
    next_value = critic(next_state.to(device)).squeeze()
    
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
