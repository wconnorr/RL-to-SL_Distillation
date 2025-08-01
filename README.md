[![DOI](https://zenodo.org/badge/641653727.svg)](https://doi.org/10.5281/zenodo.16658503)

# RL-to-SL Distillation
## Distilling Reinforcement Learning into Single-Batch Datasets

This repo contains the code used in experimentation for the paper "Distilling Reinforcement Learning into Single-Batch Datasets". The scripts have been altered slightly to increase readability, but the structure largely remains the same.

For a full explanation, see the paper. For a quick TLDR: We are distilling reinforcement learning environments into compact synthetic supervised learning datasets. We test this method on ND cart-pole, Atari, and MuJoCo environments. We compress the RL learning process on these environments into small sythentic datasets that can be learned in ONE step of gradient descent.

We plan on future refactoring to make experimentation simpler and clearer. The stable version stored on Zenodo will not reflect any future refactoring: please check the [GitHub repo](https://github.com/wconnorr/RL-to-SL_Distillation) for more up-to-date versions.

The project contains two main folders: src and distilled_envs.

### src

This contains the final experiment code used in the paper. The code is split into 3 folders, one for each environment type. Each folder contains:
- An RL experiment file
- Distillation experiment files
- Validation files
- Model files - containing distiller, agent, and critic architectures
- vector_env.py - custom environment vectorization to allow multiple environment runs to be performed simultaneously

The cart-pole environment contains an N-dimensional extension of cart-pole in nd_cartpole.py. This is largely ripped from [OpenAI's cart-pole implementation](https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py), just with state extended into an N-dimensional vector.

### distilled_envs

This folder contains the weights of the final distilled datasets. For each experiment, we have stored one of the distilled datasets, and where possible additional weights and information. Note that some are stored as .pt files (PyTorch weights) and some are stored as .ckpt files (Lightning Fabric checkpoint dictionaries). Be sure to load them using the corresponding library.

The stored distilled datasets use the following naming scheme:
- ND cart-pole - ND_cartpole/distiller_sd_\<DIMENSION\>dcp_b\<BATCH\>.pt, where DIMENSION is the N-dimensional cart-pole environment dimension, and BATCH is the number of distilled instances. Each environment has a distillation for 512 instances and one for the minimum-sized distillation. See Table 1 of the paper for results.
- Atari - Atari/\<ENV_NAME\>/\<ENCODER_LEVEL\>/distiller_sd.pt OR Atari/\<ENV_NAME\>/\<ENCODER_LEVEL\>/state_\<env_name\>.ckpt, ENCODER_LEVEL is an integer \[1,4\] showing which layer the encoder/learner split is. See Section 5.1 of the paper for further details. These are only the minimum-sized distillations, recorded in Table 2 of the paper.
- MuJoCo -  MuJoCo/\<env_name\>_b64_state.ckpt. See Table 4 of the paper for results.

The torch files (.pt) contain ONLY the distiller parameters. Additionally, where possible, we have also stored the critic parameters, encoder parameters (for l=\[1,4\]), and the optimizer parameters, allowing experiments to continue where they left off.

The checkpoint files are loaded in as dictionaries with the following keys. Unless stated otherwise, they contain the state dictionaries of the corresponding model or Adam optimizer.
- distiller
- critic
- encoder (for l=\[1,4\], not full distillation)
- outer_optimizer - state dict of the optimizer used to train the distiller
- critic_optimizer
- encoder_optimizer (if encoder exists)
- epoch - integer representing the outer epoch this state was stored at
- rewards (optional) - list of rewards achieved at the end of each episode throughout training. This was removed from longer experiments due to GitHub file size constraints.

---

### Dependencies

We provide the dependencies, including the specific versions we used to run the project in requirements.txt.

You will also need to install MuJoCo and Atari environments in gymnasium to run the RL environments. 

#### Installing Mujoco
Run `pip install gymnasium[mujoco]` after installing gymnasium.

#### Installing Atari

Atari is a bit more complex - you'll need to install them from gymnasium, but you'll also need the ROMs from ALE. Run:

`pip install gymnasium[atari]`
`pip install gymnasium[other]`

When you first import src/atari_distillation/vector_env.py, it should download the ROMs. If not, you may need to follow ALE's documentation to install the ROMs separately. If you get an error preventing you from running the hash function (seems to be OS-dependent), you may need to edit ALE's library and change their use of the MD5 hash function by adding the parameter `usedforsecurity=False` to perform the hash without throwing an error.

### Citation

If you use this work in any publication, please cite the corresponding paper.

Here is example bibtex code:
```
@inproceedings{rl-to-sl-distillation,
  title={Distilling Reinforcement Learning into Single-Batch Datasets},
  author={Wilhelm, Connor and Ventura, Dan},
  booktitle={28th European Conference on Artificial Intelligence},
  year={2025},
}
```
