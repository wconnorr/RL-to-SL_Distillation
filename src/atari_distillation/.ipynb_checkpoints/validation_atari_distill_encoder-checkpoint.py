"""
Atari Distillation w/ Encoder Validation

Since there is no outer learning, the agent is RL-algorithm agnostic. Action selection assumes network produces a policy (as in policy gradient methods like A2C and PPO), but can be changed to support DQN value-based action selection.
"""

import os
import time
import argparse

import numpy as np

import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical

from atari_models import Distiller, Actor, create_encoder_actor
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
    
  # Parser arguments
  parser = argparse.ArgumentParser()
  parser.add_argument("-b", "--inner_batch", help="inner/generated batch size", type=int, default=512)
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("-i", "--inner_epochs", help="number of inner SGD steps, using distinct batches", type=int, default=1)
  parser.add_argument("-e", "--encoder", help="if non-zero, the network will be split into an encoder and learner. Provide position of the split.", default=-1)
  parser.add_argument("-t", "--trials", help="number of trials to perform: in each, a new learner is sampled, trained on the inner task, and validated on the environment.", default=10)
  parser.add_argument("-l", "--load_from", help="load models from provided folder: will look for disiller_sd.pt and encoder_sd.pt if encoder != 0")
  parser.add_argument("--environment", help="Environment to be used, defaults to 'BreakoutNoFrameskip-v4'", default='BreakoutNoFrameskip-v4')
  parser.add_argument("result_dir", help="path to save experiment results")
  args = parser.parse_args()
  print("EXPERIMENT: ", args)
    
  global device, env_name

  env_name = args.environment
    
  if args.device:
    device = torch.device(args.device)
  else:
    device = select_device()
    print("Device not selected, using device:", device)
    
  results_path = args.result_dir

  # Using non-vectorized environment to ensure one policy is used throughout validation on a particular episode.
  env = vector_env._make_atari(env_name)()

  global n_actions
  n_actions = env.action_space.n
  init_screen, _ = env.reset()
  c, h, w = init_screen.shape
  print(env_name)
  print(n_actions, "actions")
  print("Screen size (stacked and resized):",init_screen.shape)

  ## HYPERPARAMETERS ##
  
  # INNER HYPERPARAMETERS
  # LR and momentum will be overwritten when we load in the trained distiller.
  inner_lr = 2e-2
  inner_momentum = 0
  inner_batch_size = args.inner_batch

  ## LISTS FOR PERFORMANCE ANALYSIS ##
  # OUTER STATISTICS
  end_rewards = []
    
  inner_objective = nn.MSELoss()

  # Set up distiller
  distiller = Distiller(c, h, w, n_actions, inner_batch_size, inner_lr, None, conditional_generation=False).to(device)
  if args.encoder == '0':
    encoder = None
  elif args.encoder == 'full_head':
    encoder, _ = create_encoder_actor(c, n_actions, use_full_head=True)
    encoder = encoder.to(device)
  else:
    encoder, _ = create_encoder_actor(c, n_actions, int(args.encoder))
    encoder = encoder.to(device)

  if args.load_from:
    load(distiller, os.path.join(args.load_from, "distiller_sd.pt"))
    if encoder is not None:
      load(encoder, os.path.join(args.load_from, "encoder_sd.pt"))
    
    
  begin = time.time()
  for trials in range(int(args.trials)):
    
    # Reinitialize inner network
    if encoder is None:
      actor = Actor(c, n_actions).to(device)
    elif args.encoder == 'full_head':
      actor = create_encoder_actor(c, n_actions, use_full_head=True)[1].to(device)
    else:
      actor = create_encoder_actor(c, n_actions, int(args.encoder))[1].to(device)      
    
    inner_optimizer = torch.optim.SGD(actor.parameters(), lr=distiller.inner_lr, momentum=inner_momentum)
    # Supervised inner learning (k-shot learning)
    state, actions_target = distiller()
          
    if encoder is not None:
      state = encoder(state)

    # Use actor to predict the policy/action for a given state
    actions_prediction = actor(state)
    # Classification loss: hard (CEL) w/ conditional generation, soft (MSE) w/ non-conditional
    inner_loss = inner_objective(actions_prediction, actions_target)
    # Learn on the policy network using the differentiable optimizer
    inner_loss.backward()
    inner_optimizer.step()
    inner_optimizer.zero_grad()

    state, _ = env.reset()
    cumulative_reward = 0
    done = False
    while not done:
      action = act(actor, torch.from_numpy(np.array(state)).unsqueeze(0).float().to(device), encoder)
      state, reward, done, _, _ = env.step(action)
      cumulative_reward += reward
    end_rewards.append(cumulative_reward)
  time_elapsed = time.time()-begin
  average_time = time_elapsed/int(args.trials)
  save(results_path, end_rewards, time_elapsed, average_time)
    
def save(path, rewards, time_elapsed, average_time):
    if not os.path.exists(path):
      os.makedirs(path)
    
    print("SAVING TO " + str(path))

    with open(os.path.join(path, "results.txt"), 'w') as f:
      f.write("Mean Reward: {}\nMax Reward: {}\nMin Reward: {}\nAll Rewards:\n{}\n".format(np.mean(rewards), max(rewards), min(rewards), ', '.join([str(x) for x in rewards])))
      f.write("Time Elapsed: {}\nAverage Time Per Trial: {}".format(time_elapsed, average_time))

def load(model, file_path):
    """
    model must be initialized to the correct type/size!
    file_path must point to the model's state dict .pt file!
    
    The state dict is saved on the device used in training (likely 'cuda'). map_location loads it into the current device.
    """
    model.load_state_dict(torch.load(file_path, map_location=device))
    
# Policy Gradient Methods

def act(actor, state, encoder=None):
  with torch.no_grad():
    if encoder is None:
      policy = actor(state)
    else:
      policy = actor(encoder(state))
    probs = Categorical(logits=policy)
    action = probs.sample()
  return action

if __name__ == '__main__':
  main()
