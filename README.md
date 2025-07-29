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

This folder contains the weights of the final distilled datasets. Note that some are stored as .pt files (PyTorch weights) and some are stored as .ckpt files (Lightning Fabric checkpoint dictionaries). Be sure to load them using the corresponding library.

The stored distilled datasets use the following naming scheme:
- ND cart-pole - ND_cartpole/distiller_sd_\<DIMENSION\>dcp_b\<BATCH\>.pt, where DIMENSION is the N-dimensional cart-pole environment dimension, and BATCH is the number of distilled instances. Each environment has a distillation for 512 instances and one for the minimum-sized distillation. See Table 1 of the paper for results.
- Atari - Atari/\<ENV_NAME\>/distiller_sd_\<env_name\>_<ENCODER_LEVEL>.pt OR Atari/\<ENV_NAME\>/state_\<env_name\>_<ENCODER_LEVEL>.ckpt, ENCODER_LEVEL is an integer \[1,4\] showing which layer the encoder/learner split is. See Section 5.1 of the paper for further details. These are only the minimum-sized distillations, recorded in Table 2 of the paper.
- MuJoCo -  MuJoCo/\<env_name\>_b64_state.ckpt. See Table 4 of the paper for results.

For any that have both a .pt and .ckpt file, the .ckpt was used in the final paper, while the .pt represents an earlier version.

---

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