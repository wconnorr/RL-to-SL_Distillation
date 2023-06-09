# Imports
import json
import torch

from envs import vector_env
from argparse import ArgumentParser

def main():
  with open('envs/envs.json', 'r') as json_f:
    env_names = json.loads(json_f.read())
  atari_envs = env_names['atari']
  cartpole_envs = env_names['cartpole']

  # All arguments from terminal
  parser = ArgumentParser()
  # Optional arguments:
  parser.add_argument('--rl', help='Perform RL learning or validation (no distillation).')
  parser.add_argument('--train', help='Perform training. If "--rl", performs standard RL training with a single architecture; else, performs distillation training.', action='store_true')
  parser.add_argument("--validate", help='Perform validation. If "--train", validation is performed after training is completed; otherwise, validation is performed using pretrained or randomly-initialized models.', action='store_true')
  parser.add_argument('-b', '--inner_batch_size', help='Size of inner batch: i.e. number of data instances created by the distiller.', type=int, default=512)
  parser.add_argument('-d', '--device', help="String for `torch.device` indicating which device to use for tensor computation. Defaults to cpu.", default='cpu')
  parser.add_argument("-e","--encoder", help="Layer at which to separate Actor architecture into Encoder [0,e) and Actor [e,L). If not specified, encoder will not be used. Accepts positive and negative values used as indexes into the architecture. If 0 or unspecified, no encoder is created", type=int)
  parser.add_argument('--inner_lr', help='Initial learning rate for inner optimization. This will be learned through meta-learning as a parameter in the distiller.', type=float, default=2e-2)
  # Load pretrained models / model initializations
  parser.add_argument("--load_distiller", help='Path to state dictionary file to initialize distiller. If "--rl", distiller is used to pretrain actor before RL learning.')
  parser.add_argument("--load_actor", help='Path to state dictionary file to initialize actor.')
  parser.add_argument("--load_critic", help='Path to state dictionary file to initialize critic.')
  parser.add_argument("--load_encoder", help='Path to state dictionary file to initialize encoder. Ensure "-e" value specified matches the one used to create the encoder.')
  # Save data frequencies
  parser.add_argument("--save_results_freq", help="Frequency in number of meta-epochs to save raw training result statistics. Defaults to 0, saving final results only. Does not save any training results if -1.", type=int, default=0)
  parser.add_argument("--save_models_freq", help="Frequency in number of meta-epochs to save models' parameters (Actor, Critic, and Optimizers for RL, Distiller, Critic, Encoder(optional), and Meta-Optimizer for distill). Defaults to 0, saving final models only. Does not save any results if -1.", type=int, default=0)
  parser.add_argument('--save_initial_models', help='Save model initializations for better repeatability. Saves Actor and Critic initializations for RL; Distiller, Critic, and Encoder(optional) for distill.', action='store_true')
  # Save data paths
  parser.add_argument('--save_results_dir', help='Path to folder to save raw result statistics. If not specified, defaults to current directory. Will use path to save validation results as well.')
  parser.add_argument('--save_models_dir', help='Path to folder to save model state dictionaries. If not specified, defaults to current directory.')
  

  # Required arguments:
  parser.add_argument("environment", help="Environment to be used.")
  args = parser.parse_args()
  env_name = args.environment
  use_atari = env_name in atari_envs
  use_cartpole = env_name in cartpole_envs

  # Argument-dependent local imports
  if use_atari:
    from models import atari_models as models
  elif use_cartpole:
    from models import cartpole_models as models
  else:
    raise ValueError("Environment {} not recognized. If the environment is a valid gym environment, add it to the appropriate list in ./envs/envs.json.".format(env_name))

  # Select device
  device = torch.device(args.device)
  if device.type == 'cuda':
    try:
      assert torch.cuda.is_available()
    except AssertionError:
      print("ERROR: Cannot select device {}: CUDA devices are currently unavailable.".format(device))
      exit()
  elif device.type != 'cpu':
    try:
      torch.Tensor([]).to(device)
    except Exception:
      print("ERROR: Cannot allocate tensor on device {}; device may be unavailable.".format(device))

  # Get state/action space
  env = vector_env.make_env(env_name)
  state_size = env.observation_space.shape
  action_size = env.action_space.n
  del env # we don't need the environment anymore, we just need the shapes

  # Create or load requisite models
  distiller = models.Distiller(*state_size, action_size, args.inner_batch_size, args.inner_lr)
  # TODO: Encoder
  if args.encoder:
    actor, encoder = models.create_encoder_actor(state_size[0], action_size, args.encoder)
  else:
    actor = models.Actor(state_size[0], action_size)
    encoder = None
  critic = models.Critic(state_size[0])  

  # TODO: Load in models
  if distiller is not None and args.load_distiller:
    distiller.load_state_dict(torch.load(args.load_distiller))
  if encoder is not None and args.load_encoder:
    encoder.load_state_dict(torch.load(args.load_encoder))
  if actor is not None and args.load_actor:
    actor.load_state_dict(torch.load(args.load_actor))
  if critic is not None and args.load_critic:
    critic.load_state_dict(torch.load(args.load_critic))

  # Train if train=True
  if args.train:
    pass # RL training if args.rl

  # Save trained models if specified (checkpoints every x steps)
  # Save stats if specified (checkpoints every y steps)
  # Save graphs if specified (graphed every z steps)

  # Validate (every v steps as needed)
  if args.validate:
    pass

if __name__ == '__main__':
  main()