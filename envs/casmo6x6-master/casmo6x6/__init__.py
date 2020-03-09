import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='casmo6x6-v0',
    entry_point='casmo6x6.envs:Casmo4Env',
)

