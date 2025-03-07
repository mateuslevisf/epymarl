import sys
import os
from pathlib import Path

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up to src directory
src_dir = os.path.dirname(current_dir)
# Go up to epymarl directory
epymarl_dir = os.path.dirname(src_dir)
# Go up to the parent directory containing both epymarl and mahaha
parent_dir = os.path.dirname(epymarl_dir)

# Add paths to sys.path
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Print paths for debugging
# print(f"Current directory: {os.getcwd()}")
# print(f"Looking for mahaha in: {parent_dir}")
# print(f"Python path: {sys.path}")

def env_fn(**kwargs):
    """
    Environment creation function for EPyMARL.
    Takes unpacked env_args as keyword arguments.
    """
    # Import here to avoid circular import
    from mahaha.envs.mahaha_env import MultiAgentOvercookedEnv

    # Debug what's being received
    print(f"Creating Overcooked environment with kwargs: {kwargs}")

    # Set default worker paths if not provided
    if kwargs.get("worker_a_path") is None:
        kwargs["worker_a_path"] = str(Path(parent_dir) / "agent_models_ICML/HAHA_fcp_61/worker")
    if kwargs.get("worker_b_path") is None:
        kwargs["worker_b_path"] = str(Path(parent_dir) / "agent_models_ICML/HAHA_fcp_61/worker")

    # Create and return environment
    return MultiAgentOvercookedEnv(**kwargs)