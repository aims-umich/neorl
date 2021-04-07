from neorl.rl.baselines.deepq.policies import MlpPolicy, CnnPolicy, LnMlpPolicy, LnCnnPolicy
from neorl.rl.baselines.deepq.build_graph import build_act, build_train  # noqa
from neorl.rl.baselines.deepq.dqn import DQN
from neorl.rl.baselines.shared.buffers import ReplayBuffer, PrioritizedReplayBuffer  # noqa


def wrap_atari_dqn(env):
    """
    wrap the environment in atari wrappers for DQN

    :param env: (Gym Environment) the environment
    :return: (Gym Environment) the wrapped environment
    """
    from neorl.rl.baselines.shared.atari_wrappers import wrap_deepmind
    return wrap_deepmind(env, frame_stack=True, scale=False)
