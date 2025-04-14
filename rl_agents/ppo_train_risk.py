#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from finrl.config import INDICATORS, TRAINED_MODEL_DIR
from finrl.main import check_and_make_directories
from utils.env import StockTradingEnv

check_and_make_directories([TRAINED_MODEL_DIR])

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Load dataset
train = pd.read_csv("data/train_data_deepseek_risk_2013_2018.csv")
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

# PPO agent setup
model = PPO(
    "MlpPolicy",
    env_train,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0
)

# Train the PPO agent
model.learn(total_timesteps=200000)

# Save the model
model_path = TRAINED_MODEL_DIR + "/agent_ppo_risk.pth"
model.save(model_path)
print("Training finished and saved in " + model_path)