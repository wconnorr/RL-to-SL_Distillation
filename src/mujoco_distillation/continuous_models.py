import torch
import torch.nn as nn

from torch.utils.data import Dataset


# Distiller model that parameterizes the synthetic dataset
class Distiller(nn.Module):
  def __init__(self, batch_size, state_size, action_size, inner_lr=.02, inner_momentum=0):
    super().__init__()

    self.x = nn.Parameter(torch.randn((batch_size, state_size)), True)

    self.y_mean = nn.Parameter(torch.randn((batch_size, action_size)), True)
    self.y_logstd = nn.Parameter(torch.randn(1, action_size), True)

    if inner_lr is not None:
      self.inner_lr = nn.Parameter(torch.tensor(inner_lr), True)
    if inner_momentum is not None:
      inner_momentum = nn.Parameter(torch.tensor(inner_momentum), True)

  def forward(self, dummy=None): # need dummy param for lightning
    return self.x, self.y_mean, self.y_logstd

class Encoder(nn.Module):
  def __init__(self, state_size, action_size):
    super().__init__()
    raise NotImplementedError

def create_encoder_actor(state_size, action_size):
  raise NotImplementedError

# Policy Gradient Architectures

# Return two values per action, (mu, sigma), to define the mean and std dev of a Gaussian
class Actor(nn.Module):
  def __init__(self, state_size, action_size, randomize_architecture=False): 
    super(Actor, self).__init__()
       
    hidden_size = 64

    # Produces a mean for each action based on the state
    self.mean_net = nn.Sequential(layer_init(nn.Linear(state_size, hidden_size)),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, hidden_size)),
                             nn.Tanh(),
                             layer_init(nn.Linear(hidden_size, action_size), std=.01))
    
    # Learned state-independent (log) std dev. Performance is basically the same as state-dependent std-dev
    self.logstd = nn.Parameter(torch.zeros(1, action_size))
    
  def forward(self, x):
    return self.mean_net(x.view(x.size(0),-1))

# Return a single value for a state, estimating the future discounted reward of following the current policy (it's tied to the PolicyNet it trained with)
class Critic(nn.Module):
  def __init__(self, state_size):
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
