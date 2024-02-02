import gymnasium as gym

try:
    gym.envs.register(
        id="commonroad-v1",
        entry_point="frenetix_rl.gym_environment.environment.agent_env:AgentEnv"
    )
except gym.error.Error:
    # print("[gym_commonroad/__init__.py] Error occurs while registering commonroad-v1")
    pass
