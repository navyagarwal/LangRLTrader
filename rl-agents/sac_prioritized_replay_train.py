#!/usr/bin/env python
# coding: utf-8
from datasets import load_dataset
import pandas as pd
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from utils.env import StockTradingEnv

check_and_make_directories([TRAINED_MODEL_DIR])

import numpy as np
import torch
from torch.optim import Adam
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.buffers import PrioritizedReplayBuffer

# Load dataset
# dataset = load_dataset("benstaf/nasdaq_2013_2023", data_files="train_data_deepseek_sentiment_2013_2018.csv")
# train = pd.DataFrame(dataset['train'])
train = pd.read_csv("../data/train_data_deepseek_sentiment_2013_2018.csv")
train = train.drop('Unnamed: 0', axis=1)

# Preprocess data
unique_dates = train['date'].unique()
date_to_idx = {date: idx for idx, date in enumerate(unique_dates)}
train['new_idx'] = train['date'].map(date_to_idx)
train = train.set_index('new_idx')
train['llm_sentiment'].fillna(0, inplace=True)

# Environment setup
stock_dimension = len(train.tic.unique())
state_space = 1 + 2 * stock_dimension + (1 + len(INDICATORS)) * stock_dimension
buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockTradingEnv(df=train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

# SAC agent setup with prioritized experience replay
buffer_size = 1000000
prioritized_replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6)

model = SAC('MlpPolicy', env_train, verbose=1, learning_rate=3e-4, buffer_size=buffer_size, learning_starts=1000,
            batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef='auto',
            replay_buffer=prioritized_replay_buffer)

# Train the SAC agent
model.learn(total_timesteps=200000)

# Save the model
model_path = TRAINED_MODEL_DIR + "/agent_sac_prioritized_replay.pth"
model.save(model_path)
print("Training finished and saved in " + model_path)