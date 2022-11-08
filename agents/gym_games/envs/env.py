import gym
from gym import spaces
import numpy as np
from importlib_metadata import metadata
from copy import deepcopy as cpy

gym.logger.set_level(40)


class CorDesc2dEnv(gym.Env):
    """
    Internal Environment model in the agent's mind for Cyclic coordinate descent
    optimization in a 2D setting. Here, the user model and the interface are 
    integerated together and is seen as a enviroment from agent's perspective.
    """
    #metadata = {'render.modes':['human']}

    def __init__(self, opt_game, user_model, cur):
        super().__init__()
        self.opt_game = opt_game
        self.user_model = user_model
        self.user_init_gp = cpy(user_model.gp)
        self.user_model.cur = cur
        self.opt_game.cur = cur
        self.action_space = spaces.Discrete(self.opt_game.x_arms)
        self.observation_space = spaces.Box(np.array([0, 0]), np.array([1, self.opt_game.y_arms-1]))
        self.z_queries = []

    
    def step(self, action_ai):
        """
        Update the enviroment upon agent's action
        action_ai: int
            choosen arm by the agent
        """
        self.opt_game.agent_action(action_ai)
        action_u = self.user_model.take_action(action_ai)
        self.opt_game.user_action(action_u)

        obs_u = self.opt_game.observe()
        self.user_model.update(obs_u)

        self.z_queries.append(obs_u)

        obs = (obs_u, action_u)
        #t = len(self.z_queries)
        #max_score = np.max(self.z_queries[:-1]) if len(self.z_queries)>1 else 0
        #reward = self.z_queries[-1] - (t-1)/t * max_score
        reward = np.max(self.z_queries)
        done = self.opt_game.is_done()

        return obs, reward, done, {}


    def update_usermodel(self, alpha, beta):
        """
        User model parameters and knowledge updates based on given parameters
        """
        self.user_model.reset_params(self.user_init_gp, alpha, beta)


    def reset(self):
        """
        reseting the environment
        """
        self.z_queries = []
        self.opt_game.reset()
        self.user_model.reset(self.user_init_gp)
        obs = (*self.opt_game.cur, -1)
        return obs


    def render(self, mode='human', close=False):
        """
        rendering the environment. Not implemented yet!
        """
        self.opt_game.display()
