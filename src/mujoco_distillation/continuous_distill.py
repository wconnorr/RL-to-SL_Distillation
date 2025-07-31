from lightning.fabric import Fabric

import os
import sys
import copy
import time
import random
import signal
import argparse
import pickle as pkl

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm

from torch.distributions.normal import Normal
from itertools import chain

import gymnasium as gym

import higher

from continuous_models import Distiller, Actor, Critic, RLDataset, Encoder, create_encoder_actor
import vector_env


MUJOCO_ENV_VERSION = "v4"
MUJOCO_ENV_NAMES = ['Ant', 'HalfCheetah', 'Hopper', 'Humanoid', 'HumanoidStandup', 'InvertedDoublePendulum', 'Reacher', 'Walker2d']

# Return GPU if available, else return CPU # If assert_gpu, quit the program if the GPU is not available
def select_device(assert_gpu=False):
  if assert_gpu:
    assert torch.cuda.is_available()
  return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Setting up signal to end process and save when near-termination signal is sent from SLURM
end_flag=False

def _terminate_handler(sigint, stack_frame):
  # in order to avoid having everything we want to save as a global variable and stopping mid-update, we set a flag to end the process and let it terminate naturally
  global end_flag
  end_flag=True
  with open(outfile, 'a') as f:
    f.write("Terminate signal {} received at time {}\n".format(sigint, time.time()-begin))
  return

signal.signal(signal.SIGUSR1, _terminate_handler)
save_i = 0

torch.set_float32_matmul_precision('medium') # Lowering precision for better performance

NULL_REWARD = float('inf') # we need some impossible reward value so the reward tensor contains all valid rewards

def main():
  global begin
  begin = time.time()
    
  # Parser arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--inner_batch", help="inner/generated batch size", type=int, default=512)
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'], default='cpu')
  parser.add_argument("-i", "--inner_epochs", help="number of inner SGD steps, using distinct batches", type=int, default=1)
  parser.add_argument("-p", "--policy_epochs", help="number of epochs per learning cycle", type=int, default=4)
  parser.add_argument("-r", "--reset", help="at termination, auto-resets job IF SLURM reset is set", action="store_true")
  parser.add_argument("-s", "--static_initialization", help="reinitialize parameters to a static initialization every outer iteration", action="store_true")
  parser.add_argument("-e", "--encoder", help="if non-zero, the network will be split into an encoder and learner.", action="store_true")
  parser.add_argument("-l", "--load_from", help="load models from provided folder")
  parser.add_argument("--max_epochs", help="end after performing x epochs: negative values signify infinite loop", type=int, default=-1)
  parser.add_argument("--env", help="name of MuJoCo env to distill", choices=[name + '-' + MUJOCO_ENV_VERSION for name in MUJOCO_ENV_NAMES])
  parser.add_argument("result_dir", help="path to save results plots")
  args = parser.parse_args()
  print("EXPERIMENT: ", args)
    
  fabric = Fabric(accelerator=args.device) # Fabric should use multiple devices where possible
  fabric.launch()
  env = vector_env.make_mujoco_vector_env(10, args.env)

  global device, save_i, end_flag
  device = fabric.device
  print("Process {} on GPU {}".format(fabric.global_rank,device))

  main_process = fabric.global_rank == 0

  results_path = args.result_dir

  if main_process and not os.path.exists(results_path):
    os.makedirs(results_path)
  fabric.barrier()

  global outfile
  outfile = os.path.join(results_path, "out{}.txt".format(fabric.global_rank))
  with open(outfile, 'w') as f:
    f.write("Process {} on GPU {}\n".format(fabric.global_rank,device,time.time()-begin))

  ## RL MODES ##
  use_consistent_setup = args.static_initialization

  ## HYPERPARAMETERS ##
  # OUTER HYPERPARAMETERS
  rl_lr = 2.5e-4
  num_meta_epochs = args.max_epochs
  episodes = 10 # Number of episodes to perform for each outer iteration
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
  if main_process:
    # OUTER STATISTICS
    end_rewards = []
    policy_losses = []
    value_losses = []
    entropy_losses = []

    # INNER STATISTICS
    inner_losses = []
    inner_lrs = []
  else:
    end_rewards = []
  inner_objective = nn.MSELoss()

  with open(outfile, 'a') as f:
    f.write("Hyperparameters set at time {}\n".format( time.time()-begin))

  fabric.barrier()

  action_min = float(env.action_space.low_repr)
  action_max = float(env.action_space.high_repr)
  n_actions = env.unwrapped.action_space.shape[1]
  state_size = env.unwrapped.observation_space.shape[1]

  print("Learning on {} environment with {} state values and {} actions in range[{}, {}]".format(args.env, state_size, n_actions, action_min, action_max))

  # Set up distiller
  distiller = Distiller(inner_batch_size, state_size, n_actions, inner_lr, None)
  
  outer_optimizer = optim.Adam(distiller.parameters(), lr=rl_lr, eps=adam_eps)
  distiller, outer_optimizer = fabric.setup(distiller, outer_optimizer)

  critic = Critic(state_size)
  critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr, eps=adam_eps)
  critic, critic_optimizer = fabric.setup(critic, critic_optimizer)


  if args.encoder:
    encoder, actor = create_encoder_actor(state_size, n_actions)
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=rl_lr, eps=adam_eps)
    encoder, encoder_optimizer = fabric.setup(encoder, encoder_optimizer)
  else:
    encoder, encoder_optimizer = None, None

  if use_consistent_setup:
    stable_init_sd = actor.state_dict()

  with open(outfile, 'a') as f:
    f.write("Models set at time {}\n".format( time.time()-begin))

  if args.load_from:
    rewards, meta_epoch = load_checkpoint(fabric, os.path.join(args.load_from, "state.ckpt"), distiller, critic, encoder, outer_optimizer, critic_optimizer, encoder_optimizer)
  elif args.reset and os.path.exists(os.path.join(results_path, "checkpoint1")):
    save_i = len([filename for filename in os.listdir(results_path) if "checkpoint" in filename])
    rewards, meta_epoch = load_checkpoint(fabric, os.path.join(results_path, "checkpoint{}".format(save_i), "state.ckpt"), distiller, critic, encoder, outer_optimizer, critic_optimizer, encoder_optimizer)
    if main_process:
      end_rewards = rewards
    else:
      del rewards
    with open(outfile, 'a') as f:
      f.write("Models loaded at time {}\n".format( time.time()-begin))
  else:
    meta_epoch = 0

  # Rollout data structures: constant length based on number of steps and envs.
  # No need to define these each rollout session, just overwrite them!
  states = torch.zeros((rollout_len, num_envs, state_size)).to(device)
  actions = torch.zeros((rollout_len, num_envs, n_actions)).to(device)
  prior_policy = torch.zeros((rollout_len, num_envs, n_actions)).to(device) # stored for chosen action only!
  rewards = torch.zeros((rollout_len, num_envs)).to(device)
  dones = torch.zeros((rollout_len, num_envs)).to(device)
  values = torch.zeros((rollout_len, num_envs)).to(device)
  returns = None # this will be overwritten with a new tensor, rather than being filled piecewise
  advantages = torch.zeros((rollout_len, num_envs)).to(device)
  entropies = torch.zeros((rollout_len, num_envs, n_actions)).to(device)
    
  rollout = [states, actions, prior_policy, rewards, dones, values, returns, advantages, entropies]

  initial_state, _ = env.reset()
  last_state = torch.from_numpy(initial_state).float()

  with open(outfile, 'a') as f:
    f.write("Starting learning at {}\n".format( time.time()-begin))
  while meta_epoch != num_meta_epochs and not end_flag:
    if random.randint(0, 9) == 0:
      # randomly recreate the vector environment!
      env = vector_env.make_mujoco_vector_env(10, args.env)
      initial_state, _ = env.reset()
      last_state = torch.from_numpy(initial_state).float()
    # Reinitialize inner network
    if use_consistent_setup:
      actor.load_state_dict(stable_init_sd)
    else:
      if args.encoder:
        actor = create_encoder_actor(state_size, n_actions)[1].to(device)
      else:
        actor = Actor(state_size, n_actions).to(device)
    
    init_sd = copy.deepcopy(actor.state_dict()) # this initialization will be used throughout this meta-epoch
    
    gather_data = True
    
    for policy_epoch in range(policy_epochs):
      epoch_done = False
      reset_iter = True
      memit = 0
      while not epoch_done:
        memit += 1
        
        
        distiller.inner_lr.data = torch.clamp(distiller.inner_lr.data, min=1e-6, max=1e2)
        # No need to reduce: all distiller parameters should be synced
        if main_process:
          inner_lrs.append(distiller.inner_lr.data.item())
        outer_optimizer.zero_grad()
    
        actor.load_state_dict(init_sd)
        inner_optimizer = torch.optim.SGD(actor.parameters(), lr=distiller.inner_lr,
                                        momentum=inner_momentum)

        with higher.innerloop_ctx(actor, inner_optimizer) as (h_actor, h_inner_optimizer):
          # We do not want the actors synced, so no fabric.setup here
          # Required to allow hyperparameter learning
          for param_group in h_inner_optimizer.param_groups:
            param_group['lr'] = distiller.inner_lr
    
          inner_losses_iteration = []
        
          ### SUPERVISED INNER LEARNING ###
          for inner_epoch in range(inner_epochs):
            state, action_target_mean, action_target_logstd = distiller(None)
            
            if encoder is not None:
              state = encoder(state)

            # Use actor to predict the policy/action for a given state
            action_prediction_mean = h_actor(state)
            inner_loss = inner_objective(action_prediction_mean, action_target_mean) + \
            inner_objective(h_actor.logstd, action_target_logstd)
            # Learn on the policy network using the differentiable optimizer
            h_inner_optimizer.step(inner_loss)
            
            inner_losses_iteration.append(inner_loss.item())
        
          red_inner_loss = fabric.all_reduce(np.mean(inner_losses_iteration)).item()
          if main_process:
            inner_losses.append(red_inner_loss)
        
          ### PPO OUTER TRAINING ###
          # Gather data with first net only!
          if gather_data:
            memory = []
            rollout_rewards, last_state, last_done = perform_rollout(h_actor, critic, env, rollout, rollout_len, last_state, action_min, action_max, encoder=encoder)
            general_advantage_estimation(critic, rollout, last_state, last_done, gamma, gae_lambda)
        
            gather_rollout_rewards = []
            for rank in range(fabric.world_size):
              gather_rollout_rewards.extend(fabric.broadcast(rollout_rewards, src=rank))
            if main_process:
              end_rewards.extend(gather_rollout_rewards)
            memory_dataloader = DataLoader(RLDataset(rollout), batch_size=rl_batch_size, shuffle=True)
            #memory_dataloader = fabric.setup_dataloaders(memory_dataloader)
            gather_data = False
          if reset_iter:
            memory_iter = iter(memory_dataloader)
            transition = next(memory_iter)
            reset_iter = False
    
          # Calculate and accumulate losses
          policy_loss, value_loss, entropy_loss = calculate_losses(h_actor, critic, transition, epsilon, encoder=encoder)
          loss = policy_loss + value_coefficient * value_loss - entropy_coefficient * entropy_loss
          fabric.barrier()
          fabric.backward(loss)
          fabric.barrier()
          nn.utils.clip_grad_norm_(chain(distiller.parameters(), critic.parameters()), max_grad_norm)
          outer_optimizer.step()
          outer_optimizer.zero_grad()
          if encoder is not None:
            encoder_optimizer.step()
            encoder_optimizer.zero_grad()

          critic_optimizer.step()
          critic_optimizer.zero_grad()
          red_policy_loss = fabric.all_reduce(policy_loss.item()).item()
          red_value_loss = fabric.all_reduce(value_loss.item()).item()
          red_entropy_loss = fabric.all_reduce(entropy_loss.item()).item()
          if main_process:
            policy_losses.append(red_policy_loss)
            value_losses.append(red_value_loss)
            entropy_losses.append(red_entropy_loss)
        
          try:
            transition = next(memory_iter)
          except:
            epoch_done = True
        if end_flag:
          break
      if end_flag:
        break
    meta_epoch += 1
  del env
  with open(outfile, 'a') as f: 
    f.write("Got out of main loop: hit barrier at time {}\n".format(time.time()-begin))  

  with open(outfile, 'a') as f:
    f.write("Saving checkpoint at time {}\n".format( time.time()-begin))
  save_checkpoint(fabric, os.path.join(results_path, "checkpoint" + str(save_i+1)), meta_epoch, end_rewards, distiller, critic, encoder, outer_optimizer, critic_optimizer, encoder_optimizer)
  with open(outfile, 'a') as f:
    f.write("Checkpoints saved; graphing at time {}\n".format( time.time()-begin))
  if main_process:
    graph(results_path, meta_epoch+1, end_rewards, policy_losses, value_losses, entropy_losses, inner_losses, inner_lrs)
    with open(outfile, 'a') as f:
      f.write("Graphed at time {}\n".format( time.time()-begin))
  with open(outfile, 'a') as f:
    f.write("Finished at time {}\n".format( time.time()-begin))
  fabric.barrier()
  sys.exit(0)

def save_checkpoint(fabric, path, epoch, rewards, distiller, critic, encoder, outer_optimizer, critic_optimizer, encoder_optimizer):
    fabric.barrier()
    if not os.path.exists(path) and fabric.global_rank==0:
      os.makedirs(path)
    fabric.barrier()
    state = {
      "distiller": distiller,
      "critic": critic,
      "outer_optimizer": outer_optimizer,
      "critic_optimizer": critic_optimizer,
      "rewards": rewards,
      "epoch": epoch
    }    
    if encoder is not None:
      state["encoder"] = encoder
      state["encoder_optimizer"] = encoder_optimizer
    fabric.save(os.path.join(path, 'state.ckpt'), state)

def load_checkpoint(fabric, path, distiller, critic, encoder, outer_optimizer, critic_optimizer, encoder_optimizer):
  with open(outfile, 'a') as f:
    f.write("Loading state from {} at time {}\n".format(path, time.time()-begin))

  state = fabric.load(path)
  distiller.load_state_dict(state["distiller"])
  critic.load_state_dict(state["critic"])
  outer_optimizer.load_state_dict(state["outer_optimizer"])
  critic_optimizer.load_state_dict(state["critic_optimizer"])
  if encoder is not None:
    encoder.load_state_dict(state["encoder"])
    encoder_optimizer.load_state_dict(state["encoder_optimizer"])

  with open('/'+os.path.join(os.path.join(*path.split('/')[:-1]), "distiller_{}_after.txt".format(fabric.global_rank)), 'w') as f:
    f.write("distiller:\n" + str(distiller.x) + '\n' + str(distiller.y_mean) + '\n' + str(distiller.y_logstd) + '\n' + str(distiller.inner_lr) + '\n')

  return state["rewards"], state["epoch"]

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

# Policy Gradient Methods

def act(actor, critic, state, encoder=None):
  with torch.no_grad():
    value = critic(state)
    if encoder is None:
      policy_mean = actor(state)
    else:
      policy_mean = actor(encoder(state))
    policy_std = actor.logstd.exp()
    probs = Normal(policy_mean, policy_std)
    action = probs.sample()
  return action, probs.log_prob(action), probs.entropy(), value

def calculate_losses(policy_network, value_network, transition, epsilon, encoder=None):
  states, actions, prior_policy_prob, _, _, _, returns, advantages, entropies = transition

  if encoder is None:
    current_policy = policy_network(states)
  else:
    encoded_states = encoder(states)
    current_policy = policy_network(encoded_states)
  current_policy_prob = Normal(current_policy, policy_network.logstd.exp()).log_prob(actions)

  current_values = value_network(states).squeeze(1)

  # calculate ratio quicker this way, rather than softmaxing them both
  ratio = (current_policy_prob - prior_policy_prob).exp()

  # normalize advantages
  advantages = ((advantages - advantages.mean()) / (advantages.std() + 1e-8)).unsqueeze(1)

  policy_loss = -torch.min(advantages*ratio, advantages*torch.clamp(ratio, 1-epsilon, 1+epsilon)).mean()
  value_loss = F.mse_loss(current_values, returns)
  return policy_loss, value_loss, entropies.mean()

# Performs 1 rollout of fixed length of the agent acting in the environment.
def perform_rollout(agent, critic, vec_env, rollout, rollout_len, state, action_min, action_max, encoder=None):
  with torch.no_grad():
    final_rewards = []
    states, actions, prior_policy, rewards, dones, values, _, _, entropies = rollout

    # Episode loop
    for i in range(rollout_len):
      # Agent chooses action
      action, action_distribution, entropy, value = act(agent, critic, state.to(device), encoder=encoder)

      # Env takes step based on action
      next_state, reward, term, trunc, info = vec_env.step(torch.clamp(action, action_min, action_max).cpu().numpy())

      done = np.logical_or(term, trunc)
        
      # Store step for learning
      states[i] = state
      actions[i] = action
      prior_policy[i] = action_distribution
      rewards[i] = torch.from_numpy(reward)
      dones[i] = torch.from_numpy(done)
      values[i] = value.squeeze(1)
      entropies[i] = entropy
    
      state = torch.from_numpy(next_state).float()

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
