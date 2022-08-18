import numpy as np
import gym
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optimizer

from typing import Type


class A3CAgent:
    def __init__(self, action_count):
        pass

    def actor(self, x):
        pass

    def critic(self, x):
        pass

    def train_critic(self, loss):
        pass

    def train_actor(self, loss):
        pass

class Episode:
    def __init__(self, env: gym.Env, agent: A3CAgent):
        self.env: gym.Env = env
        self.agent = agent
        self._obs = None
        self.reset()

        self.log = []

    def reset(self):
        self._obs = self.env.reset()
    
    def obs(self):
        return self._obs

    def step(self, action):
        s, r, done, info = self.env.step(action)
        self._obs = s
        if done: self.reset()
        return s, r, done, info

class A3CTrainer:
    def __init__(self, env_name: str, agent_class: Type[A3CAgent], env_count: int, max_timesteps: int, gamma = 0.99):
        self.episodes = []
        for _ in range(env_count):
            env = gym.make(env_name)
            agent = agent_class(env.action_space.n)
            episode = Episode(env, agent)
            self.episodes.append(episode)
            episode.reset()
        
        self.max_timesteps = max_timesteps
        self.gamma = gamma

    def step(self):
        for episode in self.episodes:
            s0 = episode.obs()
            a = episode.agent.actor(torch.tensor(s0).cuda())
            prob = torch.exp(a).detach().cpu().numpy()
            action = np.random.choice(range(episode.env.action_space.n))
            s1, r, done, info = episode.step(action)
            if done:
                s1 = None
            episode.log.append((s0, action, r, s1))

            if done or len(episode.log) >= self.max_timesteps:
                # Train on log
                for data in episode.log:
                    s0, action, reward, s1 = data

                    c0 = episode.agent.critic(torch.tensor(s0).cuda())
                    c1 = episode.agent.critic(torch.tensor(s1).cuda()) if s1 is not None else torch.tensor(0).cuda()
                    t = torch.tensor(reward).cuda() + self.gamma * c1.detach()
                    episode.agent.train_critic(F.mse_loss(c0, t))

                    c0 = episode.agent.critic(torch.tensor(s0).cuda())
                    advantage = t - c0.detach()
                    a = episode.agent.actor(torch.tensor(s0).cuda())
                    a_loss = -a[action] * advantage
                    episode.agent.train_actor(a_loss)

                episode.log = []


    def train(self):
        pass