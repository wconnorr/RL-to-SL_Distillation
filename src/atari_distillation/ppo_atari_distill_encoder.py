"""
Atari distillation using an encoder network.
"""

import os
import copy
import time
import argparse
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torch.distributions.categorical import Categorical
from itertools import chain

import higher

from atari_models import Distiller, Critic, RLDataset, create_encoder_actor
import vector_env

# Atari
envs = ['MsPacmanNoFrameskip-v4', 'BreakoutNoFrameskip-v4', 'CentipedeNoFrameskip-v4', 'PongNoFrameskip-v4', 'SpaceInvadersNoFrameskip-v4', 'PongNoFrameskip-v4']

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
  parser.add_argument("-b", "--inner_batch", help="inner/generated batch size", type=int, default=512)
  parser.add_argument("-c", "--conditional", help="use conditional generation (labels not learned/generated)", action="store_true")
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("-i", "--inner_epochs", help="number of inner SGD steps, using distinct batches", type=int, default=1)
  parser.add_argument("-p", "--policy_epochs", help="number of epochs per learning cycle", type=int, default=4)
  parser.add_argument("-s", "--static_initialization", help="reinitialize parameters to a static initialization every outer iteration", action="store_true")
  parser.add_argument("-t", "--timer", help="time (in minutes) to checkpoint. DOES NOT TERMINATE (for slurm restarts it needs to be auto terminated)", type=int, default=1435)
  parser.add_argument("--state_initialization", help="load real states for initializing the distiller from this file")
  parser.add_argument("-e", "--encoder", help="if non-zero, the network will be split into an encoder and learner. Provide position of the split.", default=-1)
  parser.add_argument("-l", "--load_from", help="load models from provided folder: will look for disiller_sd.pt and critic_sd.pt")
  parser.add_argument("--max_epochs", help="end after performing x epochs: negative values signify infinite loop", type=int, default=-1)
  parser.add_argument("--environment", help="Environment to be used, defaults to 'BreakoutNoFrameskip-v4'", default='BreakoutNoFrameskip-v4')
  parser.add_argument("result_dir", help="path to save results plots")
  args = parser.parse_args()
  print("EXPERIMENT: ", args)
    
  time_limit = args.timer * 60

  global device, env_name

  env_name = args.environment
    
  if args.device:
    device = torch.device(args.device)
  else:
    device = select_device()
    print("Device not selected, using device:", device)
    
  results_path = args.result_dir

  # This env is created to get us the correct state and action space dimensions. A vectorized version is used in training.
  env = vector_env._make_atari(env_name)()

  global n_actions
  n_actions = env.action_space.n
  init_screen, _ = env.reset()
  c, h, w = init_screen.shape
  print(env_name)
  print(n_actions, "actions")
  print("Screen size (stacked and resized):",init_screen.shape)


  ## RL MODES ##
#   randomize_inner_architecture = args.randomize_inner_architecture
  use_consistent_setup = args.static_initialization

  ## GENERATOR MODES ##
  use_conditional_generation = args.conditional

  ## HYPERPARAMETERS ##
  # OUTER HYPERPARAMETERS
  rl_lr = 2.5e-4
  num_meta_epochs = args.max_epochs
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

  # CRITIC HYPERPARAMETERS
  critic_lr = 2.5e-4
  
  # INNER HYPERPARAMETERS
  inner_lr = 2e-2
  inner_momentum = 0
  inner_epochs = args.inner_epochs
  inner_batch_size = args.inner_batch

  ## LISTS FOR PERFORMANCE ANALYSIS ##
  # OUTER STATISTICS
  end_rewards = []
  policy_losses = []
  value_losses = []
  entropy_losses = []

  # INNER STATISTICS
  inner_losses = []
  inner_lrs = []
    
  inner_objective = nn.CrossEntropyLoss() if use_conditional_generation else nn.MSELoss()

  env = vector_env.make_atari_vector_env(num_envs, env_name)

  # Set up distiller
  distiller = Distiller(c, h, w, n_actions, inner_batch_size, inner_lr, None, conditional_generation=use_conditional_generation).to(device)
  if args.state_initialization:
    states = torch.load(args.state_initialization)
    num_states = states.size(0)
    selected_states = states[torch.multinomial(torch.ones(num_states), inner_batch_size, replacement=(num_states<inner_batch_size)), :]
    with torch.no_grad():
      distiller.x.data = selected_states
      distiller.to(device)
    
  # Set static targets when using conditional generation
  if use_conditional_generation:
    actions_target = torch.randn(0, n_actions, (inner_batch_size,), device=device)
    actions_target_one_hot = F.one_hot(actions_target, num_classes=n_actions)

  outer_optimizer = optim.Adam(distiller.parameters(), lr=rl_lr, eps=adam_eps)

  critic = Critic(c).to(device)
    
  critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr, eps=adam_eps)

  encoder, actor = create_encoder_actor(c, n_actions, int(args.encoder))
  encoder = encoder.to(device)
  actor = actor.to(device)
  encoder_optimizer = optim.Adam(encoder.parameters(), lr=rl_lr, eps=adam_eps)


  if use_consistent_setup:
    stable_init_sd = actor.state_dict()

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

  initial_state, _ = env.reset()
  last_state = torch.from_numpy(initial_state)

  meta_epoch = 0

  if args.load_from and os.path.exists(args.load_from):
    load(distiller, os.path.join(args.load_from, "distiller_sd.pt"))
    outer_optimizer = optim.Adam(distiller.parameters(), lr=rl_lr,     eps=adam_eps) 
    load(critic,    os.path.join(args.load_from, "critic_sd.pt"      ))
    critic_optimizer = optim.Adam(critic.parameters(),   lr=critic_lr, eps=adam_eps)
    load(encoder, os.path.join(args.load_from, "encoder_sd.pt"))
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=rl_lr, eps=adam_eps)
    with open(os.path.join(args.load_from, "reward.pkl"), 'rb') as f:
      end_rewards = pkl.load(f)
    if os.path.exists(os.path.join(args.load_from, "etc.txt")):
      with open(os.path.join(args.load_from, "etc.txt"), 'r') as f:
        for line in f.readlines():
          key, value = line.split(':')
          if key == "Epoch":
            meta_epoch = int(value)

  while(meta_epoch != num_meta_epochs):
    
    # Reinitialize inner network
    if use_consistent_setup:
      actor.load_state_dict(stable_init_sd)
    else:
      if args.encoder == 'full_head':
        actor = create_encoder_actor(c, n_actions, use_full_head=True)[1].to(device)
      else:
        actor = create_encoder_actor(c, n_actions, int(args.encoder))[1].to(device)      
    
    init_sd = copy.deepcopy(actor.state_dict()) # this initialization will be used throughout this meta-epoch
    
    gather_data = True
    
    for policy_epoch in range(policy_epochs):
      epoch_done = False
      reset_iter = True
      memit = 0
      while not epoch_done:
        memit += 1
        
        
        distiller.inner_lr.data = torch.clamp(distiller.inner_lr.data, min=1e-6)
        inner_lrs.append(distiller.inner_lr.data.item())
        outer_optimizer.zero_grad()
    
        actor.load_state_dict(init_sd)
        inner_optimizer = torch.optim.SGD(actor.parameters(), lr=distiller.inner_lr,
                                        momentum=inner_momentum)

        with higher.innerloop_ctx(actor, inner_optimizer) as (h_actor, h_inner_optimizer):
          # Required to allow hyperparameter learning
          for param_group in h_inner_optimizer.param_groups:
            param_group['lr'] = distiller.inner_lr
    
          inner_losses_iteration = []
        
          ### SUPERVISED INNER LEARNING ###
          for inner_epoch in range(inner_epochs):
            if use_conditional_generation:
              state = distiller()
            else:
              state, actions_target = distiller()
            
            if encoder is not None:
              state = encoder(state)

            # Use actor to predict the policy/action for a given state
            actions_prediction = h_actor(state)
            # Classification loss: hard (CEL) w/ conditional generation, soft (MSE) w/ non-conditional
            inner_loss = inner_objective(actions_prediction, actions_target)
            # Learn on the policy network using the differentiable optimizer
            h_inner_optimizer.step(inner_loss)
            
            inner_losses_iteration.append(inner_loss.item())
        
          inner_losses.append(np.mean(inner_losses_iteration))
        
          ### PPO OUTER TRAINING ###
          # Gather data with first net only!
          if gather_data:
            rollout_rewards, last_state, last_done = perform_rollout(h_actor, critic, env, rollout, rollout_len, last_state, encoder=encoder)
            
            general_advantage_estimation(critic, rollout, last_state, last_done, gamma, gae_lambda)
        
            end_rewards.extend(rollout_rewards)
            memory_dataloader = DataLoader(RLDataset(rollout), batch_size=rl_batch_size, shuffle=True)
            gather_data = False
            
          if reset_iter:
            memory_iter = iter(memory_dataloader)
            transition = next(memory_iter)
            reset_iter = False
    
          # Calculate and accumulate losses
          policy_loss, value_loss, entropy_loss = calculate_losses(h_actor, critic, transition, epsilon, encoder=encoder)
          loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_loss
          loss.backward()
          nn.utils.clip_grad_norm_(chain(distiller.parameters(), critic.parameters()), max_grad_norm)
          outer_optimizer.step()
          outer_optimizer.zero_grad()
          encoder_optimizer.step()
          encoder_optimizer.zero_grad()

          critic_optimizer.step()
          critic_optimizer.zero_grad()
        
          policy_losses.append(policy_loss.item())
          value_losses.append(value_loss.item())
          entropy_losses.append(entropy_loss.item())
        
          try:
            transition = next(memory_iter)
          except:
            epoch_done = True
    
    if meta_epoch % 5000 == 0:
      graph(results_path, meta_epoch, end_rewards, policy_losses, value_losses, entropy_losses, inner_losses, inner_lrs)

    meta_epoch += 1
    
    if time.time()-begin > time_limit:
      save_checkpoint(os.path.join(results_path, "checkpoint_final"), meta_epoch, end_rewards, distiller.state_dict(), critic.state_dict(), encoder.state_dict(), outer_optimizer.state_dict(), critic_optimizer.state_dict(), encoder_optimizer.state_dict())
      quit()

def save_checkpoint(path, epoch, rewards, distiller_sd, critic_sd, encoder_sd, outer_optimizer_sd, critic_optimizer_sd, encoder_optimizer_sd):
    if not os.path.exists(path):
      os.makedirs(path)
    torch.save(distiller_sd, os.path.join(path, "distiller_sd.pt"))
    torch.save(outer_optimizer_sd, os.path.join(path, "outer_optimizer_sd.pt"))
    torch.save(critic_sd, os.path.join(path, "critic_sd.pt"))
    torch.save(critic_optimizer_sd, os.path.join(path, "critic_optimizer_sd.pt"))
    if encoder_sd is not None:
      torch.save(encoder_sd, os.path.join(path, "encoder_sd.pt"))
      torch.save(encoder_optimizer_sd, os.path.join(path, "encoder_optimizer_sd.pt"))
    with open(os.path.join(path, "reward.pkl"), 'wb') as f:
      pkl.dump(rewards, f)
    with open(os.path.join(path, "etc.txt"), 'w') as f:
      f.write("Epoch:{}\n".format(epoch))    


def graph(path, epoch, rewards, policy_losses, value_losses, entropy_losses, inner_losses, inner_lrs):
    path = os.path.join(path, str(epoch))
    if not os.path.exists(path):
      os.makedirs(path)
    
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
    plt.xlabel("Outer Optimization Step")
    plt.ylabel("PPO Policy Loss")
    plt.title("Outer Policy Losses")
    fig.savefig(os.path.join(path, "outer_policy_loss.png"), dpi=fig.dpi)

    fig = plt.figure()
    plt.plot(value_losses)
    plt.plot([e for e in pd.Series.rolling(pd.Series(value_losses), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(value_losses), 100).mean()])
    plt.xlabel("Outer Optimization Step")
    plt.ylabel("PPO Value Loss")
    plt.title("Value Losses")
    fig.savefig(os.path.join(path, "value_loss.png"), dpi=fig.dpi)
    
    fig = plt.figure()
    plt.plot(entropy_losses)
    plt.plot([e for e in pd.Series.rolling(pd.Series(entropy_losses), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(entropy_losses), 100).mean()])
    plt.xlabel("Outer Optimization Step")
    plt.ylabel("Policy Entropy Loss")
    plt.title("Entropy Losses")
    fig.savefig(os.path.join(path, "entropy_loss.png"), dpi=fig.dpi)
    
    fig = plt.figure()
    plt.plot(inner_losses)
    plt.plot([e for e in pd.Series.rolling(pd.Series(inner_losses), 10).mean()])
    plt.plot([e for e in pd.Series.rolling(pd.Series(inner_losses), 100).mean()])
    plt.xlabel("Outer Optimization Step")
    plt.ylabel("Supervised Loss")
    plt.title("Inner Losses")
    fig.savefig(os.path.join(path, "inner_loss.png"), dpi=fig.dpi)

    fig = plt.figure()
    plt.plot(inner_lrs)
    plt.xlabel("Outer Optimization Step")
    plt.ylabel("SGD Learning Rate")
    plt.title("Inner Supervised Learning Rate")
    fig.savefig(os.path.join(path, "inner_lr.png"), dpi=fig.dpi)

    plt.close('all')

def load(distiller, file_path):
    """
    distiller must be initialized to the correct type/size!
    file_path must point to the distiller's state dict .pt file!
    
    The state dict is saved on the device used to train the distiller that run. map_location loads it into the currently-used training device!
    """
    distiller.load_state_dict(torch.load(file_path, map_location=device))
    
# Policy Gradient Methods

def act(actor, critic, state, encoder=None):
  with torch.no_grad():
    value = critic(state)
    if encoder is None:
      policy = actor(state)
    else:
      policy = actor(encoder(state))
    probs = Categorical(logits=policy)
    action = probs.sample()
  return action, probs.log_prob(action), probs.entropy(), value

def calculate_losses(policy_network, value_network, transition, epsilon, encoder=None):
  states, actions, prior_policy, _, _, _, returns, advantages, entropies = transition

  if encoder is None:
    current_policy = policy_network(states)[F.one_hot(actions.long(), n_actions).bool()]
  else:
    current_policy = policy_network(encoder(states))[F.one_hot(actions.long(), n_actions).bool()]
  current_values = value_network(states).squeeze(1)

  # calculate ratio quicker this way, rather than softmaxing them both
  ratio = (current_policy - prior_policy).exp()

  # normalize advantages
  advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-epsilon, 1+epsilon)).mean()
  # TODO: Try clipping value loss and see if it works
  value_loss = F.mse_loss(current_values, returns)

  return policy_loss, value_loss, entropies.mean()

# Performs 1 rollout of fixed length of the agent acting in the environment.
def perform_rollout(agent, critic, vec_env, rollout, rollout_len, state, encoder=None):
  with torch.no_grad():
    final_rewards = []
    states, actions, prior_policy, rewards, dones, values, _, _, entropies = rollout

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, action_distribution, entropy, value = act(agent, critic, state.to(device), encoder=encoder)

      # Env takes step based on action
      next_state, reward, done, _, info = vec_env.step(action.cpu().numpy())

      # Store step for learning
      states[i] = state
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
          final_rewards.append(int(item['episode']['r'])) # REWARDS TURNED TO INTEGERS FOR CHEAPER CHECKPOINTING
    
  return final_rewards, state, done # no need to return rollout, its updated in-place

# Calculates advantage and return, bootstrapping using value when environment has not terminated
# Modifies them in-place in the rollout
def general_advantage_estimation(critic, rollout, next_state, next_done, gamma, gae_lambda):
  _, _, _, rewards, dones, values, returns, advantages, _ = rollout
  rollout_len = rewards.size(0)

  with torch.no_grad():
    next_value = critic(next_state.to(device)).squeeze()
    last_lambda = 0
    
    nextnonterminal = 1. - torch.from_numpy(next_done).float().to(device)
    nextvalues = next_value
    delta = rewards[rollout_len-1] + gamma * nextvalues*nextnonterminal - values[rollout_len-1]
    advantages[rollout_len-1] = last_lambda = delta # + (gamma * gae_lambda * nextnonterminal * last_lambda = 0 at iteration 0), so we can leave this part out
    for t in reversed(range(rollout_len-1)):
      nextnonterminal = 1.0 - dones[t+1]
      nextvalues = values[t+1]
      delta = rewards[t] + gamma * nextvalues*nextnonterminal - values[t]
      advantages[t] = last_lambda = delta + gamma * gae_lambda * nextnonterminal * last_lambda
    returns = advantages + values
    rollout[6] = returns
    
if __name__ == '__main__':
  main()
