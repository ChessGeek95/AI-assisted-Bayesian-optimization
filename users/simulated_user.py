from copy import deepcopy as cpy
import numpy as np


"""
Should be reimplemented this way:

class User:
    def __init__(self):
        pass

class GreedyUser(User):
    pass

class PlanningUser(User):
    pass
"""


class GreedyUser:
    def __init__(self, init_gp, n_arms, cur, theta_u):
        self.alpha = theta_u[0] # belief updating param
        self.beta = theta_u[1] # UBC search param
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T #shape(x_arms*y_arms, 2)
        self.cur = cur
        self.xy_queries = []
        self.z_queries = []

    
    def __fit_gp(self):
        """
        fitting the GP again with updated set of observed points and prior data
        """
        xy = np.append(self.xy_p, self.xy_queries).reshape(-1,2)
        z = np.append(self.z_p, self.z_queries)
        self.gp.fit(xy, z)

    
    def take_action(self, agent_action, eps=0):
        """
        choosing y for the given x in self.cur to query
        eps: int
            controlling the exploration-exploitation trade-off
        """
        x, y = np.meshgrid(agent_action, np.arange(self.y_arms))
        points = np.vstack((x.flatten(), y.flatten())).T
        mu, var = self.current_prediction(points)
        #var = var.diagonal()
        ubc_scores = mu + self.beta * np.sqrt(np.abs(var))
        action = np.argmax(ubc_scores)
        self.cur = (agent_action, action)
        self.xy_queries.append(self.cur)
        return action

    
    def update(self, obs):
        """
        update the knowledge (GP) using new observation
        """
        psudo_obs = self.__bias_point(obs)
        self.z_queries.append(psudo_obs)
        self.__fit_gp()

    
    def __bias_point(self, z):
        """
        Calculating the psudo point for the update
        """
        mu = self.current_prediction(return_std=False)
        biased_z = self.alpha * mu + (1-self.alpha) * z
        return biased_z

    
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

    
    def test(self):
        pass

    
    def reset(self, data):
        self.xy_queries = []
        self.z_queries = []
        self.xy_p, self.z_p = data
        self.__fit_gp()
        # there should be more