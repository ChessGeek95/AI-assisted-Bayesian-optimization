import matplotlib
from sklearn.preprocessing import scale
matplotlib.use('Agg')

from pickletools import read_uint1
from typing import overload

import matplotlib.pyplot as plt
import numpy as np

class OptGame:
    """
    Optimization Game class, a helper class for CorDesc2dEnv class
    """
    
    def __init__(self, function_df, function_std, n_arms, max_steps=10):
        """
        function: numpy array
            of size n_arms, representing the 2D function for the optimization task
        n_arms: tuple(int, int)
            showing the number of actions for the agent and the user, respectively
        max_steps: int
            maximum number of iterations for optimizing the function
        """
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.cur = None
        self.max_steps = max_steps
        self.function_df = np.array(function_df)#[:self.x_arms][:self.y_arms])
        self.function_std = function_std
        #self.function = function
        self.queries = []

    
    def _query(self, points):
        """
        returns evaluations of the points upon the function

        points: a tuple or a numpy array of shape (N, 2)
            the points to be evaluated
        """
        if type(points) == tuple:
            mu = self.function_df[points[1], points[0]]
            sig = self.function_std[points[1], points[0]]
            return np.random.normal(loc=mu, scale=sig)
            #mu, sig = self.gp.predict(points, return_cov=True)
            #sample = np.random.normal(loc=mu, scale=sig)
            #return sample
        else: 
            return self.function[points[:,1], points[:,0]]
            #mu, _ = self.gp.predict(points, return_cov=True)
            #return mu
    
    
    def agent_action(self, x):
        """
        Applies the agent action

        x: int
            Agent action
        """
        self.cur = (x, self.cur[1])
        #self.queries.append(self.cur)

    
    def user_action(self, y):
        """
        Applies the user action

        y: int
            User action
        """
        self.cur = (self.cur[0], y)
        self.queries.append(self.cur)

    
    def observe(self):
        """
        Provides observation upon action
        """
        return self._query(self.cur)

    
    def reset(self):
        """
        Resets the game
        """
        self.queries = []

    
    def is_done(self):
        """
        Returns True if the game finishes
        """
        return len(self.queries) >= self.max_steps
    
    
    def save(self):
        pass

    
    def display(self, animated=False):
        pass
