# HyGen: Hybrid Training for Enhanced Multi-task Generalization in MARL

[Paper Link](https://openreview.net/forum?id=53FyUAdP7d&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2023%2FConference%2FAuthors%23your-submissions))

This is the implementation of the paper "Hybrid Training for Enhanced Multi-task Generalization in Multi-agent Reinforcement Learning". 

## Installation instructions

### Install StarCraft II

Set up StarCraft II and SMAC:

```bash
bash install_sc2.sh
```

This will download SC2.4.10 into the 3rdparty folder and copy the maps necessary to run over. You may also need to persist the environment variable `SC2PATH` (e.g., append this command to `.bashrc`):

```bash
export SC2PATH=[Your SC2 folder like /abc/xyz/3rdparty/StarCraftII]
```

### Install Python environment

Install Python environment with conda:

```bash
conda create -n HyGen python=3.10 -y
conda activate HyGen
pip install -r requirements.txt
```

### Configure the SMAC package

We require additional maps from the original [SMAC](https://github.com/oxwhirl/smac) for multi-task evaluation. Below is a straightforward script that modifies 'smac' and copies these additional maps into the StarCraft II installation. Please ensure that the 'SC2PATH' is configured correctly.

```bash
git clone https://github.com/oxwhirl/smac.git
pip install -e smac/
bash install_smac_patch.sh
```

## Run experiments

You can execute the following command to run HyGen with a toy task config, which will perform training on a small batch of data:

```bash
python src/main.py --mto --config=odis --env-config=sc2_offline --task-config=toy --seed=1
```

The `--task-config` flag can be followed with any existing config name in the `src/config/tasks/` directory, and any other config named `xx` can be passed by `--xx=value`. 

As the dataset is large, we only contain the toy task config of `3m` medium data in the `dataset` folder from the default code base. Therefore, we provide the data link to the full dataset by this [Google Drive URL](https://drive.google.com/file/d/1yyqMBwZkEV6SIXB7F41Lc9tQeCoq_Nza/view?usp=sharing) and you can substitute the original data with the full dataset. After putting the full dataset in the `dataset` folder, you can run experiments in our pre-defined task sets like 

```bash
python src/main.py --mto --config=odis --env-config=sc2_offline --task-config=marine-hard-expert --seed=1
```

All results will be stored in the `results` folder and visualization results will be presented in wandb.

```bash
wandb login
```

Provide your wandb API key when prompted. (Get one from https://wandb.com)
## References

- [ODIS](https://github.com/LAMDA-RL/ODIS)
- [PyMARL](https://github.com/oxwhirl/pymarl)
- [SMAC](https://github.com/oxwhirl/smac)
  
## License

Code licensed under the Apache License v2.0.
