from neorl.rl.baselines.deepq.experiments.train_cartpole import main as train_cartpole
from neorl.rl.baselines.deepq.experiments.enjoy_cartpole import main as enjoy_cartpole
from neorl.rl.baselines.deepq.experiments.train_mountaincar import main as train_mountaincar
from neorl.rl.baselines.deepq.experiments.enjoy_mountaincar import main as enjoy_mountaincar
import os
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)

class DummyObject(object):
    """
    Dummy object to create fake Parsed Arguments object
    """
    pass


args = DummyObject()
args.no_render = True
args.max_timesteps = 200


def test_cartpole():
    train_cartpole(args)
    enjoy_cartpole(args)
    os.remove('cartpole_model.zip')

def test_mountaincar():
    train_mountaincar(args)
    enjoy_mountaincar(args)
    os.remove('mountaincar_model.zip')
