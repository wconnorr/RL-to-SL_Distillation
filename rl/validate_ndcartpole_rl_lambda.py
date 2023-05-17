import os
import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.distributions.categorical import Categorical
from itertools import chain


from models.cartpole_validation_models import Actor, Actor_XE, Actor_Variable, Critic, RLDataset
from envs import vector_env

n_actions = None
state_size = None
num_envs = 10 # number of parallel environments to gather data from
device = None

# Return GPU if available, else return CPU
# If assert_gpu, quit the program if the GPU is not available
def select_device(assert_gpu=False):
  if assert_gpu:
    assert torch.cuda.is_available()
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
  global n_actions, state_size, device
    
  # Parser arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-a", "--agent_architecture", help="architectures of agent learners", choices=['lambda','xe','random_size','random_hidden'])
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("-p", "--policy_epochs", help="number of epochs per learning cycle", type=int, default=4)
  parser.add_argument("-e", "--epochs", help="number of epochs", type=int, default=10000)
  parser.add_argument("-t", "--trials", type=int, default=20)
  parser.add_argument("--save_models", help="save models in results-dir", action="store_true")
  parser.add_argument("--filename", help="name of save file", default="out.txt")
  parser.add_argument("result_dir", help="path to save results plots")
  parser.add_argument("--degrees_of_freedom", help="degrees of freedom for nd cartpole environment: standard 2D cartpole has 1 dof", type=int, default=1)
  args = parser.parse_args()
  print("EXPERIMENT: ", args)
    
  if args.device:
    device = torch.device(args.device)
  else:
    device = select_device()
    print("Device not selected, using device:", device)
    
  results_path = args.result_dir

  ## RL MODES ##
  ## HYPERPARAMETERS ##
  # OUTER HYPERPARAMETERS
  lr = 2.5e-4
  num_epochs = args.epochs
  rollout_len = 200 # number of steps to take in the envs per rollout
  policy_epochs = args.policy_epochs
  rl_batch_size = 512 # Batch size can work with a wide range of values
  gamma = .99
  gae_lambda = .95
  epsilon = .2
    
  max_grad_norm = .5

  entropy_coefficient = .01
  value_coefficient = .5
    
  adam_eps = 1e-5 # default parameter is 1e-8

  reward_threshold = 500

  ## LISTS FOR PERFORMANCE ANALYSIS ##
  # OUTER STATISTICS
  trial_rewards = []

  for _ in range(args.trials):  
    end_rewards = []

    if args.degrees_of_freedom > 1:
      env = vector_env.make_ndcartpole_vector_env(num_envs, args.degrees_of_freedom, reward_threshold)
    else:
      env = vector_env.make_cartpole_vector_env(num_envs)

    n_actions = 2 * args.degrees_of_freedom
    state_size = 4 * args.degrees_of_freedom
    
    if args.agent_architecture == 'lambda':
      policy_network = Actor(state_size, n_actions).to(device)
    elif args.agent_architecture == "xe":
      policy_network = Actor_XE(state_size, n_actions).to(device)
    elif args.agent_architecture == "random_size":
      policy_network = Actor_Variable(state_size, n_actions, n_hiddens=random.randint(0,5)).to(device)
    elif args.agent_architecture == "random_hidden":
      policy_network = Actor(state_size, n_actions, hidden_size=random.randint(32,256)).to(device)

    value_network = Critic(state_size).to(device)
   
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=lr, eps=adam_eps)
    value_optimizer = optim.Adam(value_network.parameters(), lr=lr, eps=adam_eps)

    # Rollout data structures: constant length based on number of steps and envs.
    # No need to define these each rollout session, just overwrite them!
    states = torch.zeros((rollout_len, num_envs, state_size)).to(device)
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
    
    last_state = torch.from_numpy(env.reset()[0])
    
    for epoch in range(num_epochs):
    
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
          policy_optimizer.step()
          policy_optimizer.zero_grad()
          value_optimizer.step()
          value_optimizer.zero_grad()
        
          step += 1
    

          try:
            transition = next(memory_iter)
          except:
            epoch_done = True
    trial_rewards.append(end_rewards[-1])
  save(results_path, args.filename, trial_rewards, args)

def save(path, filename, rewards, args):
  if not os.path.exists(path):
    os.makedirs(path)
  with open(os.path.join(path, filename), 'w') as f:
    f.write("EXPERIMENT: " + str(args) + '\n\n')
    f.write(str(rewards) + "\n")
    f.write("\tAVG:" + str(np.mean(rewards)) + "\tMAX:" + str(max(rewards)) + "\n")

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
  current_values = value_network(states)

  # calculate ratio quicker this way, rather than softmaxing them both
  ratio = (current_policy - prior_policy).exp()

  # normalize advantages
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-epsilon, 1+epsilon)).mean()
  # TODO: Try clipping value loss and see if it works
  value_loss = F.mse_loss(current_values.squeeze(1), returns)
    
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
      next_state, reward, done, _, info = vec_env.step(action.cpu().numpy())

      # Store step for learning
      states[i] = state.view(num_envs, state_size) # flatten each state observation
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
