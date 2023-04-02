from copy import deepcopy as cpy
import numpy as np
"""
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS
"""
from agents.ba_mcts_v2 import BAMCTS_v2
from agents.ba_mcts import BAMCTS, BASPW, BADPW
from agents.gym_games.envs import CorDesc2dEnv
from agents.gym_games.envs.opt_game import OptGame
from agents.usermodels import UserModel
import utils
from agents import approx_bayes

import warnings
#warnings.filterwarnings("ignore")


class BaseAgent:
    """
    Base AI agent class to be inherited by various types of agent classes
    """
    def __init__(self, init_gp, n_arms, cur, max_iters):
        """
        init_gp: GaussianProcess
        n_arms: tuple(int, int)
        cur: tupe(int, int)
            the current position in the optimization task
        max_iters: int
            maximum iterations for the task
        """
        self.name = "Random"
        self.init_gp = cpy(init_gp)
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.cur = cur
        self.max_iters = max_iters
        self.user_data_is_known = False
        self.xy_queries = []
        self.z_queries = []

    def __fit_gp(self):
        """
        fitting the GP again with updated set of observed points and prior data
        """
        xy = np.append(self.xy_p, self.xy_queries).reshape(-1,2) #### check if it works correctly
        z = np.append(self.z_p, self.z_queries)
        if xy.size > 0:
            self.gp.fit(xy, z)

    def take_action(self):
        """
        Choosing a random y for the given x in self.cur to query
        """
        action = np.random.choice(np.arange(self.x_arms))
        self.cur = (action, self.cur[1])
        #self.xy_queries.append(self.cur)
        return action

    def update(self, obs, user_action):
        """
        update the knowledge (GP) using new observation
        """
        self.cur = (self.cur[0], user_action)
        self.xy_queries.append(self.cur)
        self.z_queries.append(obs)
        self.__fit_gp()

    def current_prediction(self, points=None, cur=True, return_std=True, return_cov=False):
        """
        Computing the current prediction for the given point based on the GP
        """
        if points is None:
            if cur: # current point
                res = self.gp.predict(np.array(self.cur).reshape(1,-1), return_std=return_std, return_cov=return_cov)
            else: # all points in the 2D space
                res = self.gp.predict(self.all_points, return_std=return_std, return_cov=return_cov)
        else: # the given points
            res = self.gp.predict(points, return_std=return_std, return_cov=return_cov)
        return res
    
    def posterior_samples(self, n_samples=50000):
        """
        drawing n_samples of the 2D space from the posterior GP
        """
        z_samples = self.gp_model.sample_y(self.all_points, n_samples)
        return z_samples

    def reset(self, data, user_data=None):
        self.gp = cpy(self.init_gp)
        self.xy_queries = []
        self.z_queries = []
        if len(data) == 0:
            self.xy_p = np.array([])
            self.z_p = np.array([])
        else:
            self.xy_p, self.z_p = data
        #print(np.array(self.xy_p).shape)
        if user_data:
            self.user_data_is_known = True
            self.xy_u, self.z_u = user_data
        self.__fit_gp()
        





class EpsGreedyAgent(BaseAgent):
    def __init__(self, init_gp, n_arms, cur, max_iters, beta):
        super().__init__(init_gp, n_arms, cur, max_iters)
        self.name = "Greedy"
        self.beta = beta

    def take_action(self):
        """
        choosing y for the given x in self.cur to query
        """
        #x, y = np.meshgrid(np.arange(self.x_arms), self.cur[1])
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        points = np.vstack((x.flatten(), y.flatten())).T
        mu, std = self.current_prediction(points)
        ucb_scores = utils.eval_ucb_scores(mu, std, self.beta)
        action = np.argmax(ucb_scores)
        x_idx = action % self.y_arms
        #y_idx = action // self.y_arms
        self.cur = (x_idx, self.cur[1])
        return x_idx

        """
        x, y = np.meshgrid(np.arange(self.x_arms), self.cur[1])
        points = np.vstack((x.flatten(), y.flatten())).T
        action = np.argmax(self.current_prediction(points)[0]) # choosing the point with highest mu
        self.cur = (action, self.cur[1])
        return action
        """
        


class StrategicAgent(BaseAgent):
    def __init__(self, init_gp, n_arms, cur, max_iters, user_params=None):
        super().__init__(init_gp, n_arms, cur, max_iters)
        self.name = "Planning"
        self.theta_u = user_params
        self.user_actions = []

    def take_action(self):
        """
        Choosing tha action following two steps:
            - Estimating alpha and beta distributions => Current method is ABC
            - Planning by adaptive-MCP using the estimated parameters
        """
        N_SIMS = int(2000 + 500 * len(self.z_queries)/self.max_iters)

        if self.theta_u: # Known user params
            alpha_u = np.repeat([self.theta_u[0]], N_SIMS)
            beta_u = np.repeat([self.theta_u[1]], N_SIMS)
        else: # unknown user params
            if len(self.xy_queries) == 0:
                alpha_u = np.random.random(size=N_SIMS)
                beta_u = np.random.random(size=N_SIMS)
            else:
                alpha_u, beta_u = self.params_posterior_samples(N_SIMS)
                
        # clipping to (0,1)
        alpha_u = np.clip(alpha_u, 0, 1)
        beta_u = np.clip(beta_u, 0, 1) 
        theta_u = (alpha_u, beta_u)

        # planning -> sample alpha and beta, then run MCTS, finally evaluate the actions

        if self.user_data_is_known: ####### user_data are considered to be known ##########
            user_model = UserModel(self.init_gp, 
                                (self.x_arms, self.y_arms), 
                                (self.xy_queries, self.z_queries),
                                (self.xy_u, self.z_u)) 
        else:
            user_model = UserModel(self.init_gp, 
                                (self.x_arms, self.y_arms), 
                                (self.xy_queries, self.z_queries))

        function_df, function_std = self.current_prediction(cur=False)
        function_df = function_df.reshape((self.y_arms, self.x_arms))
        function_std = function_std.reshape((self.y_arms, self.x_arms))
        #print("$"*40, function_df.shape)
        #********************************************
        #n_steps = self.max_iters-len(self.z_queries)
        #if n_steps > 2: n_steps = 2

        #n_steps = 1
        
        n_steps = 1 + int(3 * len(self.z_queries)/self.max_iters)
        if n_steps > self.max_iters-len(self.z_queries):
            n_steps = self.max_iters-len(self.z_queries)
        #********************************************
        opt_game = OptGame(function_df, function_std, #cpy(self.gp),
                            (self.x_arms, self.y_arms),
                            max_steps=n_steps)
        env = CorDesc2dEnv(opt_game, user_model, self.cur)
        y_cur = env.reset()[1]
        #model = BADPW(alpha=0.3, beta=0.2, theta_u=theta_u, initial_obs=y_cur, env=env, K=3**0.5)
        #model = BASPW(alpha=0.3, theta_u=theta_u, initial_obs=y_cur, env=env, K=3**0.5)
        #model = BAMCTS(theta_u, initial_obs=y_cur, env=env, K=2**0.5)
        model = BAMCTS_v2(theta_u, initial_obs=y_cur, env=env, K=2**0.5)
        model.learn(N_SIMS, progress_bar=False)
        action = model.best_action()
        self.cur = (action, self.cur[1])
        #print("#"*30, action)
        #self.xy_queries.append(self.cur)
        return action


    def update(self, obs, user_action):
        """
        Updating AI policy and knowledge as well as the user model
        """
        super().update(obs, user_action)
        self.user_actions.append(user_action)
        # fit the user model based on how accurate it predicted the user's action


    def params_posterior_samples(self, n_samples):
        if self.user_data_is_known: ####### user_data are considered to be known ##########
            data = [(self.xy_queries, self.z_queries, self.xy_u, self.z_u)]
        else:
            data = [(self.xy_queries, self.z_queries, [], [])]
        y_arms = self.y_arms
        alpha_u, beta_u = approx_bayes.draw_posterior_samples(n_samples, data, y_arms)
        return alpha_u, beta_u
        

