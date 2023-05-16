"""
K-shot (1-shot?) learning validation using a fully-trained distiller.
The distiller will not be trained here; rather, it will be used to teach a variety of in-distribution models.
"""

import os
import time
import random
import argparse

import numpy as np

import torch
import torch.nn as nn


from torch.distributions.categorical import Categorical

import gym

from models.cartpole_validation_models import Distiller, Actor, Actor_Ortho1, Actor_XE, Actor_Variable
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
  parser.add_argument("-a", "--agent_architecture", help="architectures of agent learners", choices=['lambda','ortho1','xe','random_size','random_hidden'])
  parser.add_argument("-b", "--inner_batch", help="inner/generated batch size", type=int, default=512)
  parser.add_argument("-d", "--device", help="select device", choices=['cuda', 'cpu'])
  parser.add_argument("-i", "--inner_epochs", help="number of inner SGD steps, using distinct batches", type=int, default=1)
  parser.add_argument("-l", "--num_learners", help="number of separately initialized learners to verify", type=int, default=1)
  parser.add_argument("-t", "--num_trials", help="number of RL trails to run for each learner after training: required for randomness in action selection and in environment behavior", type=int, default=1)
  parser.add_argument("--load_from", help="load models from provided folder: will look for disiller_sd.pt")
  parser.add_argument("--degrees_of_freedom", help="degrees of freedom for nd cartpole environment. Standard (2d) cart pole has 1 degree of freedom. Value must be an integer > 0.", type=int, default=1)
  parser.add_argument("--filename", help="name of save file", default="out.txt")
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
  filename = args.filename

  ## RL MODES ##
#   randomize_inner_architecture = args.randomize_inner_architecture
  
  ## HYPERPARAMETERS ##
  
  # INNER HYPERPARAMETERS
  inner_lr = 2e-2
  inner_momentum = 0
  inner_epochs = args.inner_epochs
  inner_batch_size = args.inner_batch

  reward_threshold = 500 # max running reward

  # TODO: Find best validation measures

  inner_objective = nn.MSELoss()

  if args.degrees_of_freedom <= 1:
    env = gym.make("CartPole-v1") 
  else:
    env = RewardTerminationWrapper(NDCartPoleEnv(args.degrees_of_freedom), reward_threshold)
    STATE_SPACE  = 4*args.degrees_of_freedom
    ACTION_SPACE = 2*args.degrees_of_freedom

  # Set up distiller
  distiller = Distiller(inner_batch_size, STATE_SPACE, ACTION_SPACE, inner_lr, None, conditional_generation=False).to(device)
    
  load(distiller, os.path.join(args.load_from, "distiller_sd.pt"))

  num_learners = args.num_learners
  num_trials = args.num_trials
 
  rewards = [[None] * num_trials for _ in range(num_learners)]
   
  for learner_i in range(num_learners):
    
    # Reinitialize inner network
    if args.agent_architecture == "lambda":
      actor = Actor(state_size=STATE_SPACE, action_size=ACTION_SPACE).to(device)
    elif args.agent_architecture == 'ortho1':
      actor = Actor_Ortho1(state_size=STATE_SPACE, action_size=ACTION_SPACE).to(device)
    elif args.agent_architecture == "xe":
      actor = Actor_XE(state_size=STATE_SPACE, action_size=ACTION_SPACE).to(device)
    elif args.agent_architecture == "random_size":
      actor = Actor_Variable(state_size=STATE_SPACE, action_size=ACTION_SPACE, n_hiddens=random.randint(0,5)).to(device)
    elif args.agent_architecture == "random_hidden":
      actor = Actor(state_size=STATE_SPACE, action_size=ACTION_SPACE, hidden_size=random.randint(32,256)).to(device)

    inner_optimizer = torch.optim.SGD(actor.parameters(), lr=distiller.inner_lr, momentum=inner_momentum)
    ### SUPERVISED INNER LEARNING ###
    for inner_epoch in range(inner_epochs):
      inner_optimizer.zero_grad()
      # Generate a batch of data instances (and labels if necessary)
      state, actions_target = distiller()

      # Use actor to predict the policy/action for a given state
      actions_prediction = actor(state)
      # Classification loss: hard (CEL) w/ conditional generation, soft (MSE) w/ non-conditional
      inner_loss = inner_objective(actions_prediction, actions_target)
      inner_loss.backward()
      inner_optimizer.step()
    for trial in range(num_trials): 
      # Now that the learner has been trained, we can validate it on the environmenti
      with torch.no_grad():
        state, _ = env.reset()
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        term = False
        trunc = False
        cumulative_reward = 0
        while not term and not trunc:
          action = act(actor, state)
          state, reward, term, trunc, _ = env.step(action.item())
          state = torch.from_numpy(state).float().unsqueeze(0).to(device)
          cumulative_reward += reward
        rewards[learner_i][trial] = cumulative_reward

  # TODO
  # The only output we really need is a list (over the learners) of lists (over the trials) of rewards
  # What about removing randomness (at least in ACT)?
  save(results_path, filename, rewards, args)

def save(path, filename, rewards, args):
    if not os.path.exists(path):
      os.makedirs(path)
    with open(os.path.join(path, filename), 'w') as f:
      f.write("EXPERIMENT: " + str(args) + '\n\n')
      f.write(str(rewards) + "\n")
      f.write("\tAVG:" + str(np.mean(rewards)) + "\tMAX:" + str(max(rewards)) + "\n")


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

if __name__ == '__main__':
  main()
