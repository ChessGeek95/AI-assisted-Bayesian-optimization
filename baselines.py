import numpy as np
from copy import deepcopy as cpy
import utils


class UBC:
    def __init__(self, function_df, init_gp, n_arms, user_data, agent_data):
        self.function_df = function_df
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.xy_p = user_data
        self.xy_p = np.append(self.xy_p, agent_data).reshape(-1,2)
        #print(user_data)
        #print(agent_data)
        #print(self.xy_p)
        self.z_p = self.__query(self.xy_p)
        self.xy_queries = []
        self.z_queries = []
        self.__fit_gp()


    def run(self, max_iter):
        for _ in range(max_iter):
            self.take_action()
            z = self.__query()
            self.z_queries.append(z)
            self.__fit_gp()
        z_tmp = np.maximum.accumulate(self.z_queries)
        f_max = np.max(self.function_df)
        scores = z_tmp/f_max * 100
        return scores


    def __query(self, points=None):
        if points is None:
            return self.function_df[self.cur[1], self.cur[0]]
        else: 
            return self.function_df[points[:,1], points[:,0]]

    
    def __fit_gp(self):
        """
        fitting the GP again with updated set of observed points and prior data
        """
        xy = np.append(self.xy_p, self.xy_queries).reshape(-1,2)
        z = np.append(self.z_p, self.z_queries)
        self.gp.fit(xy, z)

    
    def take_action(self, beta=0.2):
        """
        choosing y for the given x in self.cur to query
        eps: int
            controlling the exploration-exploitation trade-off
        """
        points_idx = np.random.choice(np.arange(len(self.all_points)), size=(100,), replace=False)
        points = self.all_points[points_idx]
        mu, std = self.current_prediction(points)
        ucb_scores = utils.eval_ucb_scores(mu, std, beta)
        self.cur = points[np.argmax(ucb_scores)]        
        self.xy_queries.append(self.cur)

    
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