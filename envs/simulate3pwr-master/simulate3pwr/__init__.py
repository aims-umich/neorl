import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='simulate3pwr-v0',
    entry_point='simulate3pwr.envs:SIMULcolEnv3',
)

