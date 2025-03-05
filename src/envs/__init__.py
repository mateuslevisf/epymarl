import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper
from .overcooked import env_fn as overcooked_fn

def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}
REGISTRY["gymma"] = gymma_fn
REGISTRY["overcooked"] = overcooked_fn
