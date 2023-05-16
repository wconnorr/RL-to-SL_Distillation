import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader


# ACTION SPACE: steer cart [left, right]
ACTION_SPACE = 2
# STATE SPACE:  [x_cart, vx_cart, theta_pole_, vtheta_pole] (not necessarily in that order)
STATE_SPACE  = 4

# Generator Architectures
# Distillation-style wrapper to learn the teaching data directly.
class Distiller(nn.Module):
  def __init__(self, batch_size, state_size=STATE_SPACE, action_size=ACTION_SPACE, inner_lr=.02, inner_momentum=.5, conditional_generation=True):
    super(Distiller, self).__init__()
    self.conditional_generation = conditional_generation
    
    x = torch.randn((batch_size, state_size))
    
    self.x = nn.Parameter(x, True)
    if not conditional_generation:
      self.y = nn.Parameter(torch.randn((batch_size, action_size)), True)

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


# Policy Gradient Architectures

# Inner Network! Return a value for each action, determining the stochastic policy for a given state.
class Actor(nn.Module):
  def __init__(self, state_size=STATE_SPACE, action_size=ACTION_SPACE, hidden_size=64):
    super(Actor, self).__init__()

    # Note: Weight norm does not help Cartpole Distillation!!!
    self.net = nn.Sequential(layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, action_size), std=.01))
    
    
  def forward(self, x):
    return self.net(x.view(x.size(0),-1))


# Inner Network! Return a value for each action, determining the stochastic policy for a given state.
class Actor_Ortho1(nn.Module):
  def __init__(self, state_size=STATE_SPACE, action_size=ACTION_SPACE, hidden_size=64):
    super(Actor_Ortho1, self).__init__()

    # Note: Weight norm does not help Cartpole Distillation!!!
    self.net = nn.Sequential(layer_init(nn.Linear(state_size, hidden_size),1),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, hidden_size),1),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, action_size), std=1))


  def forward(self, x):
    return self.net(x.view(x.size(0),-1))

# Inner Network! Return a value for each action, determining the stochastic policy for a given state.
class Actor_XE(nn.Module):
  def __init__(self, state_size=STATE_SPACE, action_size=ACTION_SPACE):
    super(Actor_XE, self).__init__()

    hidden_size = 64

    # Note: Weight norm does not help Cartpole Distillation!!!
    self.net = nn.Sequential(layer_init_xe(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             layer_init_xe(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             layer_init_xe(nn.Linear(hidden_size, action_size), std=.01))


  def forward(self, x):
    return self.net(x.view(x.size(0),-1))

class Actor_Variable(nn.Module):
  def __init__(self, state_size=STATE_SPACE, action_size=ACTION_SPACE, n_hiddens=0):
    super(Actor_Variable, self).__init__()
    
    hidden_size = 64

    layers = [layer_init(nn.Linear(state_size, hidden_size))] + [layer_init(nn.Linear(hidden_size, hidden_size)) for _ in range(n_hiddens)] + [layer_init(nn.Linear(hidden_size, action_size), std=.01)]

    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x.view(x.size(0),-1))

# Return a single value for a state, estimating the future discounted reward of following the current policy (it's tied to the PolicyNet it trained with)
class Critic(nn.Module):
  def __init__(self, state_size=STATE_SPACE):
    super(Critic, self).__init__()
    
    hidden_size = 64
  
    self.net = nn.Sequential(layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, 1), std=1.))
    
  def forward(self, x):
    return self.net(x.view(x.size(0),-1))

# Dataset that wraps memory for a dataloader
class RLDataset(Dataset):
  def __init__(self, rollout):
    super().__init__()
    self.rollout = rollout
    self.width = len(rollout) # Number of distinct value types saved (state, action, etc.)
    self.rollout_len, self.num_envs,_ = rollout[0].shape
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

def layer_init_xe(layer, std=ROOT_2, bias_const=0.0):
  torch.nn.init.xavier_normal_(layer.weight, gain=std)
  torch.nn.init.constant_(layer.bias, bias_const)
  return layer

# Splits the actor after encoder_size layers: the first half is the encoder, the second half is the remaining actor.
def create_encoder_actor(state_size=STATE_SPACE, action_size=ACTION_SPACE, encoder_size=-1):
  encoder = Actor(state_size, action_size)
  actor = encoder.net[encoder_size:]
  encoder.net = encoder.net[:encoder_size]
  return encoder, actor
