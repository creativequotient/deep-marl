import yaml
import pathlib
import os
import sys
import logging
import time
import numpy as np
import torch as T

from wrappers import REGISTRY as env_registry
from controllers.basic_controller import BasicController


def remove_dim(tensor_list):
    """
    Converts outputs from tensors to numpy arrays and removes a
    dimension, useful for converting network outputs which have a
    batch dim to actions
    """
    result = []
    for tensor in tensor_list:
        result.append(tensor.detach().cpu().numpy())
    return list(map(lambda x: x[0], result))

def run(configs):
    experimental_dir = configs['exp_dir']
    pathlib.Path(experimental_dir).mkdir(parents=True, exist_ok=True)

    # Initialize environment
    env = env_registry[configs['env']['scenario']]
    n_agents, obs_shape_n, state_shape, act_shape_n, act_type = env.info()

    # Controller
    controller = BasicController(obs_shape_n, act_shape_n, state_shape, act_type, configs)

    # Training parameters
    max_episodes = configs['exp']['num_episodes']
    update_interval = configs['learner']['update_interval']
    max_episode_len = configs['env']['episode_len']

    # Reward and stat tracking
    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(n_agents)]

    # Begin training
    timestep = 0
    t_start = time.time()
    for episode in range(1, max_episodes + 1):
        obs_n, state = env.reset()
        avail_actions = env.get_avail_actions()
        for episode_step in range(max_episode_len):
            # Increment total timesteps
            timestep += 1

            # Get actions from controller
            with T.no_grad():
                act_n = remove_dim(controller.step(obs_n, avail_actions, explore=True))

            next_obs_n, rew_n, done_n, next_state = env.step(act_n)
            next_avail_actions = env.get_avail_actions()

            # Add to experience
            controller.experience(act_n=act_n,
                                  avail_n=avail_actions,
                                  obs_n=obs_n,
                                  next_obs_n=next_obs_n,
                                  next_avail_n=next_avail_actions,
                                  rew_n=rew_n,
                                  done_n=done_n,
                                  state=state,
                                  next_state=next_state)

            # Train if possible
            if timestep % update_interval == 0:
                controller.update()

            # Update obs_n
            obs_n = next_obs_n
            state = next_state
            avail_actions = next_avail_actions

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            # End episode if done
            if all(done_n):
                break

        episode_rewards.append(0)
        for a in agent_rewards:
            a.append(0)

        if episode % 1000 == 0:
            # get past 1000 steps
            print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    timestep, len(episode_rewards) - 1, np.mean(episode_rewards[-1000:]),
                    round(time.time() - t_start, 3)))
            t_start = time.time()
