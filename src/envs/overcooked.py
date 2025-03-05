from mahaha.envs.mahaha_env import MultiAgentOvercookedEnv
from .multiagentenv import MultiAgentEnv

def env_fn(env_args, **kwargs) -> MultiAgentEnv:
    return MultiAgentOvercookedEnv(**env_args)