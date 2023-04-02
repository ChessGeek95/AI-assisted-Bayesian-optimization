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
        if len(user_data) == 0:
            self.xy_p = np.array(agent_data).reshape(-1,2)
            self.z_p = self.__query(self.xy_p)
            if len(agent_data) == 0:
                self.xy_p = np.array([])
                self.z_p = np.array([])
        elif len(agent_data) == 0:
                self.xy_p = np.array(user_data).reshape(-1,2)
                self.z_p = self.__query(self.xy_p)
        else:
            self.xy_p = np.append(user_data, agent_data).reshape(-1,2)
            self.z_p = self.__query(self.xy_p)
        self.xy_queries = []
        self.z_queries = []
        self.__fit_gp()


    def run(self, max_iter, top_k=1):
        for _ in range(max_iter):
            self.take_action(top_k)
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
        elif len(points) == 0:
            return None
        else: 
            return self.function_df[points[:,1], points[:,0]]

    
    def __fit_gp(self):
        """
        fitting the GP again with updated set of observed points and prior data
        """
        xy = np.append(self.xy_p, self.xy_queries).reshape(-1,2)
        z = np.append(self.z_p, self.z_queries)
        if xy.size > 0:
            self.gp.fit(xy, z)

    
    def take_action(self, top_k, beta=0.2):
        """
        choosing y for the given x in self.cur to query
        eps: int
            controlling the exploration-exploitation trade-off
        """
        points_idx = np.random.choice(np.arange(len(self.all_points)), size=(100,), replace=False)
        points = self.all_points[points_idx]
        mu, std = self.current_prediction(points)
        ucb_scores = utils.eval_ucb_scores(mu, std, beta)
        if top_k < 1:
            top_k = 1
        idx_topk = np.argpartition(ucb_scores, -top_k)[-top_k:]
        idx = np.random.choice(idx_topk)
        self.cur = points[idx]
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