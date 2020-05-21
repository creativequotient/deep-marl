import os
import time

import numpy as np
import visdom

from Agents import MADDPGAgent
from Common import trainer_parse_args, make_env, make_dirs, setup_experiment_dir
from Networks import MLP

args = trainer_parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

# Work in progress
# if args.load_dir != "":
#     scenario_dir = os.path.join('experiments', args.scenario, 'MADDPG')
#     experimental_dir = setup_experiment_dir(scenario_dir)
# else:
#     experimental_dir = args.load_dir

# Set-up environment
env = make_env(args.scenario, args.benchmark)

# Set-up learners
n_agents = len(env.observation_space)
agents = [MADDPGAgent(f'Agent {i}',
                      args.local_q,
                      MLP,
                      args.num_units,
                      args.lr,
                      args.gamma,
                      args.tau,
                      args.batch_size,
                      args.batch_size * args.max_episode_len,
                      args.buffer_size,
                      args.update_every,
                      env.observation_space[i].shape,
                      (env.action_space[i].n,),
                      args.discrete,
                      env.observation_space,
                      env.action_space) for i in range(n_agents)]

if args.load_dir != "":
    for agent in agents:
        load_path = os.path.join(args.load_dir, agent.id)
        agent.load_agent(load_path)

if args.port > 0:
    viz = visdom.Visdom(port=args.port)

update_counter = 1

for episode in range(args.num_episodes):
    obs = env.reset()
    overall_reward = []
    visdom_info = {}
    for agent in agents:
        agent.reset_noise()
    for t in range(args.max_episode_len):
        actions = [agent.get_action(o, noise=True)[0].cpu().detach().numpy() for agent, o in
                   zip(agents, obs)]
        new_obs, rew, done, info = env.step(actions)
        for i, agent in enumerate(agents):
            agent.experience(obs[i], actions[i], rew[i], new_obs[i], done[i])
        if update_counter % args.update_every == 0:
            for agent in agents:
                agent.prep_training()
            for i, agent in enumerate(agents):
                info = agent.update(agents, i)
                if info is not None:
                    if agent.id in visdom_info:
                        visdom_info[agent.id]['q_loss'] += [info['q_loss'].detach().cpu().numpy()]
                        visdom_info[agent.id]['pi_loss'] += [info['pi_loss'].detach().cpu().numpy()]
                    else:
                        visdom_info[agent.id] = {}
                        visdom_info[agent.id]['q_loss'] = [info['q_loss'].detach().cpu().numpy()]
                        visdom_info[agent.id]['pi_loss'] = [info['pi_loss'].detach().cpu().numpy()]
            for agent in agents:
                agent.soft_update(agent.tau)
                agent.prep_rollouts()
        update_counter = (update_counter + 1) % args.update_every
        overall_reward.append(rew[0])
        obs = new_obs
        if args.display:
            env.render()
        if all(done):
            break
    if args.port > 0:
        viz.line(
            X=[episode],
            Y=[np.sum(overall_reward)],
            win='Reward',
            update='append',
            name='Training',
            opts=dict(title='Rewards', linecolor=np.array((0, 255, 0)).reshape(1, -1)))
        for agent_id in visdom_info:
            viz.line(
                X=[episode],
                Y=[np.average(visdom_info[agent_id]['q_loss'])],
                win=agent_id,
                name='Q-loss',
                update='append',
                opts=dict(title=f'{agent_id}', linecolor=np.array((0, 255, 0)).reshape(1, -1)))
            viz.line(
                X=[episode],
                Y=[np.average(visdom_info[agent_id]['pi_loss'])],
                win=agent_id,
                name='pi-loss',
                update='append',
                opts=dict(title=f'{agent_id}', linecolor=np.array((255, 0, 0)).reshape(1, -1)))
    if episode % args.eval_every == 0 and episode != 0:
        eval_rewards = []
        for eval_episode in range(10):
            obs = env.reset()
            overall_reward = []
            for t in range(args.max_episode_len):
                actions = [agents[i].get_action(obs[i], noise=False)[0].cpu().detach().numpy() for i in range(len(agents))]
                new_obs, rew, done, info = env.step(actions)
                overall_reward.append(rew[0])
                obs = new_obs
                if args.display:
                    env.render()
                if all(done):
                    break
            eval_rewards.append(sum(overall_reward))
        print(f'Episode: {episode} Eval reward: {np.average(eval_rewards)}')
        for agent in agents:
            save_path = os.path.join(args.save_dir, agent.id)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            agent.save_agent(save_path)
        if args.port > 0:
            viz.line(
                X=[episode],
                Y=[np.sum(overall_reward)],
                win='Reward',
                update='append',
                name='Evaluation',
                opts=dict(title='Rewards', linecolor=np.array((255, 0, 0)).reshape(1, -1)))

# Evaluate
eval_rewards = []
for eval_episode in range(100):
    obs = env.reset()
    overall_reward = []
    for t in range(args.max_episode_len):
        actions = [agents[i].get_action(obs[i])[0].cpu().detach().numpy() for i in range(len(agents))]
        new_obs, rew, done, info = env.step(actions)
        overall_reward.append(rew[0])
        obs = new_obs
        env.render()
        if all(done):
            break
    eval_rewards.append(sum(overall_reward))
print(f'Average evaluation reward: {np.average(eval_rewards)}')
print(f'Evaluation reward stdev: {np.std(eval_rewards)}')
