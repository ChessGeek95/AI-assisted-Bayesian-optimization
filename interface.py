import numpy as np

class Interface:
    """
    The actual interface class where the interaction between the agent and the actual user takes place!
    """
    def __init__(self, n_arms):
        """
        n_arms: tuple(int, int)
            number of arms on the x and y axes, respectively.
        """
        self.xy_queries = []
        self.z_queries = []
        self.cur = (0,0)
        self.n_arms = n_arms
        x, y = np.meshgrid(np.arange(self.n_arms[0]), np.arange(self.n_arms[1]))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T


    def _query(self, points):
        """
        points: a tuple or a numpy array of shape (N, 2)
        """
        if points is None or len(points) == 0:
            return None
        if type(points) == tuple:
            return self.function[points[1], points[0]]
            #return self.function[points[0], points[1]]
        else:
            return self.function[points[:,1], points[:,0]]
            #return self.function[points[:,0], points[:,1]]

    def get_cur(self):
        return self.cur

    def init_user(self):
        z = self._query(self.user_data)
        if z is None:
            z = []
        return self.user_data, z

    def init_agent(self):
        z = self._query(self.agent_data)
        if z is None:
            z = []
        return self.agent_data, z

    def step(self, x_t, y_t):
        """
        Updates the interface upon action
        x_t: int
            agent's action
        y_t: int
            user's action
        """
        self.cur = (x_t, y_t)
        #z = self.function[self.cur[0], self.cur[1]]
        z = self._query(self.cur)
        self.xy_queries.append(self.cur)
        self.z_queries.append(z)
        return z

    def reset(self, function_df, user_data, agent_data, init_point):
        """
        Resets the interface, generates another function and reinitialize parameters
        """
        self.xy_queries = []
        self.z_queries = []
        self.function = function_df
        self.user_data = user_data
        self.agent_data = agent_data
        self.start = init_point
        self.cur = init_point
        return self.init_user(), self.init_agent()
    
    def get_score(self):
        """
        Calculate the score for performing the task
        """
        func_max = np.max(self.function)
        z_max = np.max(self.z_queries)
        return np.round(z_max, 3), np.round(100*z_max/func_max, 2)

    def display(self):
        pass