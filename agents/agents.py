from copy import deepcopy as cpy
import numpy as np
import elfi
import torch
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import MCMC, NUTS

from agents.ba_mcts import BAMCTS, BASPW, BADPW
from agents.gym_games.envs import CorDesc2dEnv
from agents.gym_games.envs.opt_game import OptGame
from agents.usermodels import UserModel, UserModel_ABC, UserModel_Pyro
from agents.approx_bayes import ABC
import utils



class BaseAgent:
    """
    Base AI agent class to be inherited by various types of agent classes
    """
    def __init__(self, init_gp, n_arms, cur, max_iters=10):
        """
        init_gp: GaussianProcess
        n_arms: tuple(int, int)
        cur: tupe(int, int)
            the current position in the optimization task
        max_iters: int
            maximum iterations for the task
        """
        self.init_gp = cpy(init_gp)
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.cur = cur
        self.max_iters = max_iters
        self.xy_queries = []
        self.z_queries = []

    def __fit_gp(self):
        """
        fitting the GP again with updated set of observed points and prior data
        """
        xy = np.append(self.xy_p, self.xy_queries).reshape(-1,2) #### check if it works correctly
        z = np.append(self.z_p, self.z_queries)
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
        self.xy_queries = []
        self.z_queries = []
        self.xy_p, self.z_p = data
        if user_data:
            self.xy_u, self.z_u = user_data
        self.__fit_gp()
        


class EpsGreedyAgent(BaseAgent):
    def __init__(self, init_gp, n_arms, cur, beta=0.2):
        super().__init__(init_gp, n_arms, cur)
        self.beta = beta

    def take_action(self):
        """
        choosing y for the given x in self.cur to query
        """
        x, y = np.meshgrid(np.arange(self.x_arms), self.cur[1])
        points = np.vstack((x.flatten(), y.flatten())).T
        mu, std = self.current_prediction(points)
        ucb_scores = utils.eval_ucb_scores(mu, std, self.beta)
        action = np.argmax(ucb_scores)
        self.cur = (action, self.cur[1])
        return action

        """
        x, y = np.meshgrid(np.arange(self.x_arms), self.cur[1])
        points = np.vstack((x.flatten(), y.flatten())).T
        action = np.argmax(self.current_prediction(points)[0]) # choosing the point with highest mu
        self.cur = (action, self.cur[1])
        return action
        """
        


class StrategicAgent(BaseAgent):
    def __init__(self, init_gp, n_arms, cur):
        super().__init__(init_gp, n_arms, cur)
        self.user_actions = []


    def generative_model(self, alpha, beta, batch_size=1, random_state=None):
        """
        Simulator model for ABC
        """
        # Make inputs 2d arrays for numpy broadcasting with w
        alpha = np.asanyarray(alpha).reshape((-1, 1))
        beta = np.asanyarray(beta).reshape((-1, 1))
        
        simulated_actions = []
        user_model = UserModel_ABC(self.init_gp, 
                                (self.x_arms, self.y_arms), 
                                (self.xy_queries, self.z_queries),
                                self.cur)

        for i in range(alpha.shape[0]):
            a = alpha[i]
            b = beta[i]
            trial_actions = []
            for j in range(len(self.user_actions)):
                user_model.reset_params(self.init_gp, a, b)
                user_act = user_model.take_action()
                trial_actions.append(user_act)
            simulated_actions.append(trial_actions)

        simulated_actions = np.array(simulated_actions)
        return simulated_actions


    def params_posterior_samples(self, n_samples=1000):

        def model():
            alpha = pyro.sample("alpha", dist.Uniform(torch.zeros(1), torch.ones(1)))
            beta = pyro.sample("beta", dist.Uniform(torch.zeros(1), torch.ones(1)))
            sigma = 1.

            ############### build the model #################
            # obs = argmax_y{E[f(x,y)] + beta * Var[f(x,y)] + eps}

            ubc_list = []
            user_model = UserModel_Pyro(self.init_gp, 
                                        (self.x_arms, self.y_arms), 
                                        (self.xy_queries, self.z_queries),
                                        self.cur)
            for j in range(len(self.user_actions)+1):
                data_temp = (self.xy_queries[:j], self.z_queries[:j])
                user_model.reset_params(self.init_gp, alpha, beta, data_temp)
                user_act, ubc_scores = user_model.take_action()
                ubc_list.append(ubc_scores)

            observations = []
            for i, ubc_scr in enumerate(ubc_list):
                #print(ubc_scr)
                ubc_scr = torch.normal(torch.tensor(ubc_scr), sigma)
                #print(ubc_scr)
                y_max = torch.argmax(ubc_scr)
                observations.append(y_max) #pyro.sample("obs", dist.Normal(mu, sigma))
                # how to do the argmax?
            #################################################
            observations = torch.tensor(observations).type(torch.Tensor)
            print(ubc_scr)
            return pyro.sample("obs", dist.Delta(observations))

        def conditioned_model(model, y):
            return poutine.condition(model, data={"obs": y})()

        def main(args):
            nuts_kernel = NUTS(conditioned_model, jit_compile=args[3])
            mcmc = MCMC(
                nuts_kernel,
                num_samples=args[0],
                warmup_steps=args[1],
                num_chains=args[2],
            )
            ############### prepare the inputs ##########
            mcmc.run(model, y=self.user_actions)
            mcmc.summary(prob=0.5)
            samples = mcmc.get_samples()
            return samples

        samples = main([100,100,1, False])
        alpha_u = samples["alpha"]
        beta_u = samples["beta"]
        theta_u = (alpha_u, beta_u)
        return theta_u


    def take_action(self):
        """
        Choosing tha action following two steps:
            - Estimating alpha and beta distributions => Current method is ABC
            - Planning by adaptive-MCP using the estimated parameters
        """
        
        """
        abc = ABC(self.user_actions, self.generative_model, self.y_arms)
        abc.new_model()
        alpha_u, beta_u, means = abc.fit()
        #print("ABC means:", means)
        abc.reset_model()
        theta_u = (alpha_u, beta_u)
        del abc
        """
        
        #alpha_u = np.random.random(100)*0.1+0.35
        #beta_u = np.random.random(100)*0.05+0.1
        alpha_u = np.random.normal(0.2, 0.02, size=(3000,))
        beta_u = np.random.normal(0.2, 0.02, size=(3000,))
        alpha_u = alpha_u[(alpha_u>=0) & (alpha_u<=1)]
        beta_u = beta_u[(beta_u>=0) & (beta_u<=1)]
        theta_u = (alpha_u, beta_u)
        #theta_u = self.params_posterior_samples()

        # planning -> sample alpha and beta, then run MCTS, finally evaluate the actions
        user_model = UserModel(self.init_gp, 
                                (self.x_arms, self.y_arms), 
                                (self.xy_queries, self.z_queries),
                                (self.xy_u, self.z_u))

        function_df, function_std = self.current_prediction(cur=False)
        function_df = function_df.reshape((self.y_arms, self.x_arms))
        function_std = function_std.reshape((self.y_arms, self.x_arms))
        #print("$"*40, function_df.shape)
        n_steps = self.max_iters-len(self.z_queries)
        if n_steps > 3: n_steps = 3
        opt_game = OptGame(function_df, function_std, #cpy(self.gp),
                            (self.x_arms, self.y_arms),
                            max_steps=n_steps)
        env = CorDesc2dEnv(opt_game, user_model, self.cur)
        y_cur = env.reset()[1]
        #model = BADPW(alpha=0.3, beta=0.2, theta_u=theta_u, initial_obs=y_cur, env=env, K=3**0.5)
        #model = BASPW(alpha=0.3, theta_u=theta_u, initial_obs=y_cur, env=env, K=3**0.5)
        model = BAMCTS(theta_u, initial_obs=y_cur, env=env, K=2**0.5)
        model.learn(2000, progress_bar=False)
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



