from Agents import MADDPGAgent
from Networks import MLP
import numpy as np
from RandomProcess import OrnsteinUhlenbeckActionNoise
import visdom
import os
import time
from Common import parse_args, make_env


args = parse_args()

if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

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
                      n_agents, env.observation_space, env.action_space) for i in range(n_agents)]

if args.load_dir is not "":
    for agent in agents:
        load_path = os.path.join(args.load_dir, agent.id)
        agent.load_agent(load_path)

if args.port > 0:
    viz = visdom.Visdom(port=args.port)

update_counter = 1
noise_processes = [OrnsteinUhlenbeckActionNoise(np.zeros(agents[i].act_shape,)) for i in range(len(agents))]

for episode in range(args.num_episodes):
    obs = env.reset()
    overall_reward = []
    visdom_info = {}
    for process in noise_processes:
        process.reset()
    for t in range(args.max_episode_len):
        actions = [agents[i].get_action(obs[i])[0].cpu().detach().numpy() + noise_processes[i]() for i in range(len(agents))]
        new_obs, rew, done, info = env.step(actions)
        for i, agent in enumerate(agents):
            agent.experience(obs[i], actions[i], rew[i], new_obs[i], done[i])
        for agent in agents:
            info = agent.update(agents, update_counter)
            if info is not None:
                if agent.id in visdom_info:
                    visdom_info[agent.id]['q_loss'] += [info['q_loss'].detach().cpu().numpy()]
                    visdom_info[agent.id]['pi_loss'] += [info['pi_loss'].detach().cpu().numpy()]
                else:
                    visdom_info[agent.id] = {}
                    visdom_info[agent.id]['q_loss'] = [info['q_loss'].detach().cpu().numpy()]
                    visdom_info[agent.id]['pi_loss'] = [info['pi_loss'].detach().cpu().numpy()]
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
            update = 'append',
            name = 'Training',
            opts = dict(title='Rewards', linecolor=np.array((0,255,0)).reshape(1,-1)))
        for agent_id in visdom_info:
            viz.line(
                X=[episode],
                Y=[np.average(visdom_info[agent_id]['q_loss'])],
                win=agent_id,
                name='Q-loss',
                update='append',
                opts=dict(title=f'{agent_id}', linecolor=np.array((0,255,0)).reshape(1,-1)))
            viz.line(
                X=[episode],
                Y=[np.average(visdom_info[agent_id]['pi_loss'])],
                win=agent_id,
                name='pi-loss',
                update='append',
                opts=dict(title=f'{agent_id}', linecolor=np.array((255,0,0)).reshape(1,-1)))
    if episode % args.eval_every == 0 and episode != 0:
        eval_rewards = []
        for eval_episode in range(10):
            obs = env.reset()
            overall_reward = []

            for t in range(args.max_episode_len):
                actions = [agents[i].get_target_action(obs[i])[0].cpu().detach().numpy() for i in range(len(agents))]
                new_obs, rew, done, info = env.step(actions)
                overall_reward.append(rew[0])
                obs = new_obs
                if args.display:
                    env.render()
                if all(done):
                    break

            eval_rewards.append(sum(overall_reward))
        print(f'Eval reward: {np.average(eval_rewards)}')
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

while True:
    obs = env.reset()
    overall_reward = []
    for t in range(args.max_episode_len):
        actions = [agents[i].get_target_action(obs[i])[0].cpu().detach().numpy() for i in range(len(agents))]
        new_obs, rew, done, info = env.step(actions)
        overall_reward.append(rew[0])
        obs = new_obs
        env.render()
        time.sleep(0.02)
        if all(done):
            break
    print(sum(overall_reward))
