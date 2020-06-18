import json
import os
import pathlib
import pickle
import time

import numpy as np

from utils import trainer_parse_args, make_env, make_dirs, get_learners


def train(args):
    if not os.path.exists(args['save_dir']):
        make_dirs(args['save_dir'])

    if args['load_dir'] != "":
        args = json.load(os.path.join(args['load_dir'], 'run_info.json'))
    else:
        with open(os.path.join(args['save_dir'], 'run_info.json'), 'w') as f:
            json.dump(args, f, indent=4)

    # Set-up environment
    env = make_env(args['scenario'])

    # Set-up learners
    n_agents = len(env.observation_space)
    num_adversaries = min(env.n, args['num_adversaries'])
    agents = get_learners(env, num_adversaries, args)
    if num_adversaries > 0:
        print('Using good policy {} and adv policy {}...'.format(args['good_policy'], args['adv_policy']))
    else:
        print('Using policy {}...'.format(args['good_policy']))

    if args['load_dir'] != "":
        for agent in agents:
            load_path = os.path.join(args['load_dir'], agent.agent_name)
            agent.load_agent(load_path)

    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(n_agents)]
    obs_n = env.reset()
    episode_step = 0
    train_step = 0
    t_start = time.time()

    print('Starting iterations...')
    while True:
        # collect agent actions
        action_n = [agent.get_action(o, explore=True) for agent, o in zip(agents, obs_n)]
        # execute actions in environment
        new_obs_n, rew_n, done_n, info_n = env.step(action_n)
        episode_step += 1
        done = all(done_n)
        terminal = (episode_step >= args['max_episode_len'])    
        # collect experiences
        for i, agent in enumerate(agents):
            agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i])
        # update environment
        obs_n = new_obs_n

        for i, rew in enumerate(rew_n):
            episode_rewards[-1] += rew
            agent_rewards[i][-1] += rew

        if done or terminal:
            obs_n = env.reset()
            episode_step = 0
            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            for agent in agents:
                agent.reset_noise()

        # increment global step counter
        train_step += 1

        # render environment
        if args['display']:
            time.sleep(0.03)
            env.render()

        if train_step % args['update_interval'] == 0:
            for agent in agents:
                agent.prep_training()
            for i, agent in enumerate(agents):
                agent.update(agents, i)
            for agent in agents:
                agent.soft_update(agent.tau)
                agent.prep_roll_outs()

        if terminal and (len(episode_rewards) % args['save_rate'] == 0):
            # save agents
            for agent in agents:
                save_path = os.path.join(args['save_dir'], 'models', agent.agent_name)
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                agent.save_agent(save_path)
            # print intermediate stats
            if num_adversaries == 0:
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-args['save_rate']:]),
                    round(time.time() - t_start, 3)))
            else:
                print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-args['save_rate']:]),
                    [np.mean(rew[-args['save_rate']:]) for rew in agent_rewards], round(time.time() - t_start, 3)))
            t_start = time.time()

        if len(episode_rewards) > args['num_episodes']:
            log_dir = os.path.join(args['save_dir'], 'logs')
            pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
            with open(os.path.join(log_dir, 'overall_rewards.pkl'), 'wb') as fp:
                pickle.dump(episode_rewards, fp)
            with open(os.path.join(log_dir, 'individual_rewards.pkl'), 'wb') as fp:
                pickle.dump(agent_rewards, fp)
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            break


if __name__ == "__main__":
    args = vars(trainer_parse_args())
    train(args)
