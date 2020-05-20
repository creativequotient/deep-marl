import argparse
import os
import numpy as np
import torch as T

def trainer_parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=25000, help="number of episodes")
    parser.add_argument("--port", type=int, default=0, help="visdom port")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="mixing factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--buffer-size", type=int, default=1e6, help="replay buffer size")
    parser.add_argument("--update-every", type=int, default=100, help="update interval")
    parser.add_argument("--eval-every", type=int, default=100, help="eval interval")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--local-q", action="store_true", default=False)
    parser.add_argument("--discrete", action="store_true", default=False)
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                        help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")

    return parser.parse_args()


def eval_parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=20, help="number of episodes")
    # Checkpointing
    parser.add_argument("--load-dir", type=str, default="",
                        help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--display", action="store_true", default=False)
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
    path_ = path.split("/")
    for i, fn in enumerate(path_):
        fp = os.path.join(*path_[:i], fn)
        if not os.path.exists(fp):
            os.mkdir(fp)


def setup_experiment_dir(path):
    make_dirs(path)
    counter = 0
    while True:
        fp = os.path.join(path, f'run-{counter}')
        if not os.path.exists(fp):
            os.mkdir(fp)
            return fp
        else:
            counter += 1


def onehot_from_logits(logits):
    logits = logits.detach().cpu().numpy()
    output = np.zeros(logits.shape)
    for idx, entry in enumerate(logits):
        output[idx][np.argmax(entry)] = 1.0
    return T.tensor(output, dtype=T.float32, device=T.device('cuda' if T.cuda.is_available() else 'cpu'))