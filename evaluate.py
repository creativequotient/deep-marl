import json
import os
import pickle
import time

import numpy as np

from utils import eval_parse_args, make_env, make_dirs, get_learners


# Evaluate
def evaluate(load_dir, eval_episodes, benchmark, display):
    with open(os.path.join(load_dir, 'run_info.json'), 'r') as f:
        args = json.load(f)

    # Set-up environment
    env = make_env(args['scenario'], benchmark)

    # Set-up learners
    n_agents = len(env.observation_space)
    num_adversaries = min(env.n, args['num_adversaries'])
    agents = get_learners(env, num_adversaries, args)
    if num_adversaries > 0:
        print('Using good policy {} and adv policy {}...'.format(args['good_policy'], args['adv_policy']))
    else:
        print('Using policy {}...'.format(args['good_policy']))

    for agent in agents:
        load_path = os.path.join(load_dir, 'models', agent.agent_name)
        agent.load_agent(load_path)

    episode_rewards = [0.0]
    agent_rewards = [[0.0] for _ in range(n_agents)]
    agent_info = [[[]]]  # placeholder for benchmarking info
    obs_n = env.reset()
    episode_step = 0

    print('Starting iterations...')
    while True:
        # collect agent actions
        action_n = [agent.get_action(o, explore=False) for agent, o in zip(agents, obs_n)]
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
            agent_info.append([[]])

        # for benchmarking learned policies
        if benchmark:
            for i, info in enumerate(info_n):
                agent_info[-1][i].append(info_n['n'])
            if len(episode_rewards) > args['num_episodes'] and (done or terminal):
                log_dir = os.path.join(args['save_dir'], 'logs')
                make_dirs(log_dir)
                print('Finished benchmarking, now saving...')
                with open(os.path.join(log_dir, 'benchmark_info.pkl'), 'wb') as fp:
                    pickle.dump(agent_info[:-1], fp)
                break
            continue

        # render environment
        if display:
            time.sleep(0.03)
            env.render()

        # saves final episode reward for plotting training curve later
        if len(episode_rewards) > eval_episodes:
            print('...Finished total of {} episodes.'.format(len(episode_rewards)))
            if num_adversaries == 0:
                print("episodes: {}, mean episode reward: {}, std episode reward: {}".format(
                    len(episode_rewards), np.mean(episode_rewards), np.std(episode_rewards)))
            else:
                print("episodes: {}, mean episode reward: {}, agent episode reward: {}".format(
                    len(episode_rewards), np.mean(episode_rewards),
                    [np.mean(rew) for rew in agent_rewards],
                    [np.std(rew) for rew in agent_rewards]))
            log_dir = os.path.join(args['save_dir'], 'logs')
            with open(os.path.join(log_dir, 'eval_overall_rewards.pkl'), 'wb') as fp:
                pickle.dump(episode_rewards, fp)
            with open(os.path.join(log_dir, 'eval_individual_rewards.pkl'), 'wb') as fp:
                pickle.dump(agent_rewards, fp)
            break


if __name__ == "__main__":
    args = vars(eval_parse_args())
    evaluate(**args)
