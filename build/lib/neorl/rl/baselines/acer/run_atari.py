#!/usr/bin/env python3
import warnings

from neorl.rl.baselines.shared import logger
from neorl.rl.baselines.acer.acer import ACER
from neorl.rl.baselines.shared.policies import CnnPolicy, CnnLstmPolicy
from neorl.rl.baselines.shared.cmd_util import make_atari_env, atari_arg_parser
from neorl.rl.baselines.shared.vec_env import VecFrameStack


def train(env_id, num_timesteps, seed, policy, lr_schedule, num_cpu):
    """
    train an ACER model on atari

    :param env_id: (str) Environment ID
    :param num_timesteps: (int) The total number of samples
    :param seed: (int) The initial seed for training
    :param policy: (A2CPolicy) The policy model to use (MLP, CNN, LSTM, ...)
    :param lr_schedule: (str) The type of scheduler for the learning rate update ('linear', 'constant',
                                 'double_linear_con', 'middle_drop' or 'double_middle_drop')
    :param num_cpu: (int) The number of cpu to train on
    """
    env = VecFrameStack(make_atari_env(env_id, num_cpu, seed), 4)
    if policy == 'cnn':
        policy_fn = CnnPolicy
    elif policy == 'lstm':
        policy_fn = CnnLstmPolicy
    else:
        warnings.warn("Policy {} not implemented".format(policy))
        return

    model = ACER(policy_fn, env, lr_schedule=lr_schedule, buffer_size=5000, seed=seed)
    model.learn(total_timesteps=int(num_timesteps * 1.1))
    env.close()
    # Free memory
    del model


def main():
    """
    Runs the test
    """
    parser = atari_arg_parser()
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm'], default='cnn', help='Policy architecture')
    parser.add_argument('--lr_schedule', choices=['constant', 'linear'], default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--logdir', help='Directory for logging')
    args = parser.parse_args()
    logger.configure(args.logdir)
    train(args.env, num_timesteps=args.num_timesteps, seed=args.seed,
          policy=args.policy, lr_schedule=args.lr_schedule, num_cpu=16)


if __name__ == '__main__':
    main()
