# MiniHack UED experiments

This repository contains code for the MiniHack UED experiments. The main MiniHack repository is here: https://github.com/MiniHackPlanet/MiniHack.

## Setup 

Run the following commands to set up a conda environment with the necessary dependencies:
```
conda create -n nle python=3.8
conda activate nle
pip install -r requirements.txt

git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
cd ..

git clone https://github.com/MiniHackPlanet/MiniHack --recursive
cd nle
pip install -e ".[dev]"
cd .. 
```

## Experiments

To launch a run of the MiniHack UED experiment, run the following command:
```
python -m train \
  --xpid=minihack_ued_0 \
  --env_name=MiniHack-GoalLastAdversarial-v0 \
  --use_gae=True \
  --gamma=0.995 \
  --seed=0 \
  --recurrent_arch=lstm \
  --recurrent_agent=True \
  --recurrent_adversary_env=True \
  --recurrent_hidden_size=256 \
  --checkpoint=True \
  --lr=0.0001 \
  --num_steps=256 \
  --num_processes=2 \
  --test_num_processes=4 \
  --test_interval=10 \
  --num_env_steps=100000000 \
  --ppo_epoch=5 \
  --num_mini_batch=1 \
  --entropy_coef=0.0 \
  --adv_entropy_coef=0.005 \
  --algo=ppo \
  --ued_algo=paired \
  --log_dir='~/logs/minihack/' \
  --log_interval=1 \
  --archive_interval=3052 \
  --screenshot_interval=10 \
  --test_env_names='MiniHack-Room-15x15-v0,MiniHack-MazeWalk-9x9-v0,MiniHack-MazeWalk-15x15-v0,MiniHack-Labyrinth-Small-v0,MiniHack-Labyrinth-Big-v0' \
  --log_grad_norm=False \
  --verbose
```
