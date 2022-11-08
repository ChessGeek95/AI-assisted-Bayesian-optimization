from gym.envs.registration import register

register(
    id='CoorDesc-v1',
    entry_point='gym_games.envs:CorDesc2dEnv',
    max_episode_steps= 100,
)