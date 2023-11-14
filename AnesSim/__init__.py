from gym.envs.registration import register
from .env import *

register(
    id="AnesSim-v0",
    entry_point="AnesSim:Simulator",
    max_episode_steps=500,
)
