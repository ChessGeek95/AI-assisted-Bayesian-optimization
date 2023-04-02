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

    
    def step(self, action_ai, y_dist=None):
        """
        Update the enviroment upon agent's action
        action_ai: int
            choosen arm by the agent
        """
        ucb_scores_x = self.opt_game.agent_action(action_ai)
        #***************************************************************************
        topk_idx = np.argpartition(ucb_scores_x, -self.opt_game.y_arms//5)[-self.opt_game.y_arms//5:]
        avg_topk_ucb = np.mean(ucb_scores_x[topk_idx])
        avg_ucb_user = 0
        #f = open("out.text", "a")
        if y_dist is not None:
            """
            f.writelines("y_dist shape:" + str(y_dist.shape))
            f.writelines("ucb_scores_x shape:"+str(ucb_scores_x.shape))
            f.writelines("ucb shape:"+str((y_dist * ucb_scores_x).shape))
            f.close()
            """
            avg_ucb_user = np.mean(y_dist * ucb_scores_x)
        #***************************************************************************

        action_u = self.user_model.take_action(action_ai)
        ucb_socre = self.opt_game.user_action(action_u)

        obs_u = self.opt_game.observe(randomness=False)
        self.user_model.update(obs_u)

        self.z_queries.append(obs_u)

        obs = (obs_u, action_u)

        #***************************************************************************
        #reward = np.max(self.z_queries)
        #***************************************************************************
        #reward = avg_topk_ucb + np.max(self.z_queries)
        #***************************************************************************
        """
        t = len(self.z_queries)
        max_score = np.max(self.z_queries[:-1]) if len(self.z_queries)>1 else 0
        modified_expected_improvement = self.z_queries[-1] - (t-1)/t * max_score
        if modified_expected_improvement < 0:
            modified_expected_improvement = 0
        #reward += avg_topk_ucb
        #"""
        #***************************************************************************
        #reward = 5 * (t-1)/t * modified_expected_improvement + avg_ucb_user
        #reward = 2 * modified_expected_improvement + avg_ucb_user
        #reward = 5 * (t-1)/t * modified_expected_improvement + avg_ucb_user + avg_topk_ucb
        #reward = modified_expected_improvement + avg_ucb_user + avg_topk_ucb
        #***************************************************************************
        #reward = avg_ucb_user      #141
        #reward = avg_topk_ucb      #142
        reward = avg_ucb_user + avg_topk_ucb      #143
        

        info_dict = {}
        info_dict["ucb"] = ucb_socre

        #reward = np.max(self.z_queries+[self.user_model.observe(randomness=True)])
        done = self.opt_game.is_done()

        return obs, reward, done, info_dict

    
    def give_y_dist(self, y_dist):
        self.y_dist = y_dist


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
