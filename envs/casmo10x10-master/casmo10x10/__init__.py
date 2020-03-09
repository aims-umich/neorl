import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='casmo10x10-v0',
    entry_point='casmo10x10.envs:Casmo4Env',
)

