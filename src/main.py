"""
The main script is responsible for parsing inputs and arguments before passing them to the runner for execution
"""

import argparse
import yaml
from run import run


def parse_args():
    parser = argparse.ArgumentParser('Deep multi-agent reinforcement learning experiments for multi-agent environments')
    # Evaluation
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--eval-options', type=str, default='', help='evaluation/benchmarking options/parameters')
    # Environment parameters
    parser.add_argument('--scenario', type=str, default='', help='Desired environment to execute (ie "mpe.simple_spread" for simple_spread scenario from mpe environment')
    parser.add_argument('--scenario-options', type=str, default='', help='scenario options/parameters where applicable')
    # Learner parameters
    parser.add_argument('--learner', type=str, default='', help='reinforcement learning algorithm to use')
    parser.add_argument('--learner-options', type=str, default='', help='learner options/parameters where applicable')
    parser.add_argument('--adv_learner', type=str, default='', help='reinforcement learning algorithm to use')
    parser.add_argument('--adv-learner-options', type=str, default='', help='learner options/parameters where applicable')
    # Experimental dir
    parser.add_argument('--options', type=str, default='')
    parser.add_argument('experiment_dir', type=str, help='experimental directory')

    return parser.parse_args()


def update_params(target, source):
    result = {}
    for key in target:
        if key in source:
            result[key] = source[key]
        else:
            result[key] = target[key]
    return result


def assemble_configs(arglist):

    def str2dict(options_str):
        result = {}
        if options_str == '':
            return result
        else:
            options = options_str.split(',')
            for option in options:
                op, param = option.split('=')
                try:
                    param = int(param)
                except ValueError:
                    param = param
                result[op] = param
            return result

    configs = {
        'exp_dir': arglist.experiment_dir
    }

    with open(f'configs/eval.yaml', 'r') as f:
        defaults = yaml.load(f, yaml.SafeLoader)
        configs['eval'] = update_params(defaults, str2dict(arglist.eval_options))
        configs['eval']['eval'] = True if arglist.evaluate else False

    with open(f'configs/envs/{arglist.scenario}.yaml', 'r') as f:
        defaults = yaml.load(f, yaml.SafeLoader)
        configs['env'] = update_params(defaults, str2dict(arglist.scenario_options))

    with open(f'configs/learners/{arglist.learner}.yaml', 'r') as f:
        defaults = yaml.load(f, yaml.SafeLoader)
        configs['learner'] = update_params(defaults, str2dict(arglist.learner_options))

    with open(f'configs/defaults.yaml', 'r') as f:
        defaults = yaml.load(f, yaml.SafeLoader)
        configs['exp'] = update_params(defaults, str2dict(arglist.options))

    return configs


if __name__ == '__main__':
    args = parse_args()
    configs = assemble_configs(args)
    for key in configs:
        print(f'{key} args'.upper())
        if type(configs[key]) is dict:
            for key_ in configs[key]:
                print(f'{key_}: {configs[key][key_]}')
        else:
            print(configs[key])
        print()
    run(configs)
