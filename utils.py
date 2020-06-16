import argparse
import os

from deepmarl.algorithms.maddpg import MADDPGAgent
from deepmarl.algorithms.m3ddpg import M3DDPGAgent
from deepmarl.common.networks import MLP


def trainer_parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=25000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="mixing factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--buffer-size", type=int, default=1e6, help="replay buffer size")
    parser.add_argument("--update-interval", type=int, default=100, help="update interval")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--perturbation", type=float, default=0.0, help="perturbation factor(s)")
    parser.add_argument("--discrete", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default="/tmp/policy",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)

    output = parser.parse_args()

    return output


def eval_parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--eval-episodes", type=int, default=20,
                        help="number of evaluation episodes")
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--save-gif", action="store_true", default=False)

    return parser.parse_args()


def make_env(scenario_name, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env


def make_dirs(path):
    if path[-1] == "/":
        path = path[:-1]
    path_ = path.split("/")
    for i, fn in enumerate(path_):
        fp = os.path.join(*path_[:i], fn)
        if not os.path.exists(fp):
            os.mkdir(fp)


def get_learners(env, num_adversaries, arglist, model=MLP):
    learners = []
    if arglist['good_policy'] == 'm3ddpg' or arglist['adv_policy'] == 'm3ddpg':
        perturbation_factors = [arglist['perturbation']] * env.n
        print(perturbation_factors)
    for i in range(num_adversaries):
        if arglist['adv_policy'] == 'm3ddpg':
            learners.append(M3DDPGAgent(
                "agent_%d" % i, i, model, env.observation_space, env.action_space, arglist, perturbation_factors))
        else:
            learners.append(MADDPGAgent(
                "agent_%d" % i, i, model, env.observation_space, env.action_space, arglist,
                local_q=(arglist['adv_policy'] == 'ddpg')))
    for i in range(num_adversaries, env.n):
        if arglist['good_policy'] == 'm3ddpg':
            learners.append(M3DDPGAgent(
                "agent_%d" % i, i, model, env.observation_space, env.action_space, arglist, perturbation_factors))
        else:
            learners.append(MADDPGAgent(
                "agent_%d" % i, i, model, env.observation_space, env.action_space, arglist,
                local_q=(arglist['good_policy'] == 'ddpg')))
    return learners
