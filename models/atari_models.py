# Contains the set of architectures used in the Atari learning experiments.

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

# Generator Architectures

# GTN-style generator to produce the teaching data
class GTNGenerator(nn.Module):
  def __init__(self, state_space, action_space, z_vector_size, inner_lr=.02, inner_momentum=.5, conditional_generation=True):
    super(GTNGenerator, self).__init__()

    self.input_size = z_vector_size + (action_space if conditional_generation else 0)
    self.conditional_generation = conditional_generation
    hidden_size = z_vector_size * 2
    
    self.l1 = nn.Linear(self.input_size, hidden_size)
    self.relu = nn.ReLU()
    self.state_head = nn.Linear(hidden_size, state_space)
    if not conditional_generation:
      self.action_head = nn.Linear(hidden_size, action_space)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr),True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum),True)

  def forward(self, z, y=None):
    if self.conditional_generation:
      x = torch.cat([z,y], dim=1)
      return self.state_head(self.relu(self.l1(x)))
    else:
      body = self.relu(self.l1(z))
      return self.state_head(body), F.softmax(self.action_head(body), dim=1)
    
# Distillation-style wrapper to learn the teaching data directly.
class Distiller(nn.Module):
  def __init__(self, state_c, state_h, state_w, action_space, batch_size, inner_lr=.02, inner_momentum=.5, conditional_generation=True):
    super(Distiller, self).__init__()
    self.conditional_generation = conditional_generation
    
    self.x = nn.Parameter(torch.randn((batch_size, state_c, state_h, state_w)), True)
    if not conditional_generation:
      self.y = nn.Parameter(torch.randn((batch_size, action_space)), True)

    # Inner optimizer parameters
    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      self.inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)
    
  def forward(self):
    if self.conditional_generation:
      return self.x
    else:
      return self.x, self.y

class Encoder(nn.Module):
  def __init__(self, c, h, w, embed_size):
    super(Encoder, self).__init__()
    self.convs = nn.Sequential(nn.Conv2d(c, c*4, 3, padding=1, stride=2),
                               nn.ReLU(),
                               nn.Conv2d(c*4, c*8, 3, padding=1, stride=2),
                               nn.ReLU())
    self.linear = nn.Linear(c*h*w, embed_size)

  def forward(self, x):
    x = self.convs(x)
    x.view(x.size(0), -1)
    return self.linear(x)

# Policy Gradient Architectures
class Actor(nn.Module): # an actor-critic neural network
    def __init__(self, state_channels, num_actions, simplify=False):
        super(Actor, self).__init__()

        self.convs = nn.Sequential(
          layer_init(nn.Conv2d(state_channels, 32, 8, stride=4)),
          nn.ReLU(),
          layer_init(nn.Conv2d(32,64, 4, stride=2)),
          nn.ReLU(),
          layer_init(nn.Conv2d(64, 64, 3, stride=1)),
          nn.ReLU()
        )
        h, w = (4,6) if simplify else (7,7) 
        self.head = nn.Sequential(
          layer_init(nn.Linear(64*h*w, 512)),
          nn.ReLU(),
          layer_init(nn.Linear(512, num_actions), std=.01)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.head(x)
        
class Critic(nn.Module): # an actor-critic neural network
    def __init__(self, state_channels, simplify=False):
        super(Critic, self).__init__()

        self.convs = nn.Sequential(
          layer_init(nn.Conv2d(state_channels, 32, 8, stride=4)),
          nn.ReLU(),
          layer_init(nn.Conv2d(32,64, 4, stride=2)),
          nn.ReLU(),
          layer_init(nn.Conv2d(64, 64, 3, stride=1)),
          nn.ReLU()
        )
        h, w = (4,6) if simplify else (7,7) 
        self.head = nn.Sequential(
          layer_init(nn.Linear(64*h*w, 512)),
          nn.ReLU(),
          layer_init(nn.Linear(512, 1), std=1.)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        return self.head(x)



# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
  def __init__(self, rollout):
    super().__init__()
    self.rollout = rollout
    self.width = len(rollout) # Number of distinct value types saved (state, action, etc.)
    self.rollout_len, self.num_envs, _, _, _ = rollout[0].shape
    self.full = self.rollout_len * self.num_envs
    
  def __getitem__(self, index):
    return [self.rollout[i][index//self.num_envs][index%self.num_envs] for i in range(self.width)]
 
  def __len__(self):
    return self.full


ROOT_2 = 2.0**.5

# INITIALIZATION FUNCTIONS #
def layer_init(layer, std=ROOT_2, bias_const=0.0):
  torch.nn.init.orthogonal_(layer.weight, std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

# Splits the actor after encoder_size layers: the first half is the encoder, the second half is the remaining actor.
def create_encoder_actor(state_channels, action_size, encoder_size=-1, base_model=None):
  if base_model is None:
    encoder = Actor(state_channels, action_size)
  else:
    encoder = base_model
  if encoder_size==-2:
    actor = encoder.head
    encoder.head = nn.Identity()
  elif encoder_size == -1:
    actor = encoder.head[encoder_size:]
    encoder.head = encoder.head[:encoder_size]
  elif encoder_size < 0:
    encoder_size += 2 # pass the head
    encoder_size *= 2 # paired Conv2d and ReLUs
    actor = encoder
    encoder = actor.convs[:encoder_size]
    actor.convs = actor.convs[encoder_size:]
  else:
    encoder_size *= 2 # paired Conv2d and ReLUs
    actor = encoder
    encoder = actor.convs[:encoder_size]
    actor.convs = actor.convs[encoder_size:]
  return encoder, actor
