from copy import deepcopy as cpy
import numpy as np




class UserModel:
    def __init__(self, init_gp, n_arms, data, user_data = [[],[]]):
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.cur = None
        self.xy_p, self.z_p = data
        self.xy_queries = []
        self.z_queries = []
        self.xy_u, self.z_u = user_data

    def __fit_gp(self):
        xy = np.append(self.xy_p, self.xy_queries).reshape(-1,2)
        z = np.append(self.z_p, self.z_queries)
        #print(self.xy_u, self.z_u)
        xy = np.append(xy, self.xy_u).reshape(-1,2)
        z = np.append(z, self.z_u)
        self.gp.fit(xy, z)

    def take_action(self, ai_action, eps=0):
        """
        choosing y for the given x in self.cur to query
        """
        x, y = np.meshgrid(ai_action, np.arange(self.y_arms))
        points = np.vstack((x.flatten(), y.flatten())).T
        mu, var = self.current_prediction(points)
        var = var.diagonal()
        mu = mu.reshape(-1,)
        ubc_scores = mu + self.beta * var
        action = np.argmax(ubc_scores)
        self.cur = (ai_action, action)
        self.xy_queries.append(self.cur)
        return action

    def update(self, obs):
        biased_obs = self.__bias_point(obs)
        self.z_queries.append(biased_obs)
        self.__fit_gp()

    def current_prediction(self, points=None, cur=True):
        if points is None:
            if cur:
                mu, Cov = self.gp.predict(np.array(self.cur).reshape(1,-1), return_cov=True)
            else:
                mu, Cov = self.gp.predict(self.all_points, return_cov=True)
        else:
            mu, Cov = self.gp.predict(points, return_cov=True)
        return mu, Cov

    def posterior_samples(self, n_samples=50000):
        z_samples = self.gp_model.sample_y(self.all_points, n_samples)
        return z_samples

    def __bias_point(self, z):
        mu, _ = self.current_prediction()
        biased_z = self.alpha * mu + (1-self.alpha) * z
        return biased_z


    def reset_params(self, init_gp, alpha, beta):
        self.gp = cpy(init_gp)
        self.alpha = alpha  # belief updating param
        self.beta = beta    # UBC search param
        self.xy_queries = []
        self.z_queries = []
        #"""
        biased_z = []
        for i in range(len(self.z_p)):
            self.cur = self.xy_p[i]
            biased_mu = self.__bias_point(self.z_p[i])
            biased_z.append(biased_mu)
            self.gp.fit(self.xy_p[:i+1], biased_z)
        #"""
        if len(biased_z) > 0:
            self.z_p = biased_z 
            self.__fit_gp()


    def reset(self, init_gp):
        self.gp = cpy(init_gp)
        self.alpha = None  # belief updating param
        self.beta = None    # UBC search param
        self.xy_queries = []
        self.z_queries = []






class UserModel_ABC:
    """
    User model calss for ABC simulation part
    """
    def __init__(self, init_gp, n_arms, data, cur):
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.cur = cur
        self.xy_p, self.z_p = data
        self.xy_queries = []
        self.z_queries = []

    def take_action(self, eps=0):
        
        #choosing y for the given x in self.cur to query
        
        if np.random.rand() < eps:
            action = np.random.choice(np.arange(self.y_arms))
        else:
            x, y = np.meshgrid(self.cur[0], np.arange(self.y_arms))
            points = np.vstack((x.flatten(), y.flatten())).T
            mu, var = self.current_prediction(points)
            var = var.diagonal()
            ubc_scores = mu + 2 * self.beta * np.sqrt(np.abs(var))
            action = np.argmax(ubc_scores)
        self.cur = (self.cur[0], action)
        self.xy_queries.append(self.cur)
        return action

    def __bias_point(self, z):
        mu, _ = self.current_prediction()
        biased_z = self.alpha * mu + (1-self.alpha) * z
        return biased_z


    def reset_params(self, init_gp, alpha, beta):
        self.gp = cpy(init_gp)
        self.alpha = alpha  # belief updating param
        self.beta = beta    # UBC search param
        self.xy_queries = []
        self.z_queries = []
        biased_z = []
        for i in range(len(self.z_p)):
            self.cur = self.xy_p[i]
            biased_mu = self.__bias_point(self.z_p[i])
            biased_z.append(biased_mu)
            self.gp.fit(self.xy_p[:i+1], biased_z)

    def current_prediction(self, points=None, cur=True):
        if points is None:
            if cur:
                mu, Cov = self.gp.predict(np.array(self.cur).reshape(1,-1), return_cov=True)
            else:
                mu, Cov = self.gp.predict(self.all_points, return_cov=True)
        else:
            mu, Cov = self.gp.predict(points, return_cov=True)
        return mu, Cov














class UserModel_Pyro:
    """
    User model calss for ABC simulation part
    """
    def __init__(self, init_gp, n_arms, data, cur):
        self.gp = cpy(init_gp)
        self.x_arms, self.y_arms = n_arms
        x, y = np.meshgrid(np.arange(self.x_arms), np.arange(self.y_arms))
        self.all_points = np.vstack((x.flatten(), y.flatten())).T
        self.cur = cur
        self.xy_p, self.z_p = data
        self.xy_queries = []
        self.z_queries = []

    def take_action(self, eps=0):
        
        #choosing y for the given x in self.cur to query
        
        if np.random.rand() < eps:
            action = np.random.choice(np.arange(self.y_arms))
        else:
            x, y = np.meshgrid(self.cur[0], np.arange(self.y_arms))
            points = np.vstack((x.flatten(), y.flatten())).T
            mu, var = self.current_prediction(points)
            var = var.diagonal()
            ubc_scores = mu + 2 * self.beta * np.sqrt(np.abs(var))
            action = np.argmax(ubc_scores)
        self.cur = (self.cur[0], action)
        self.xy_queries.append(self.cur)
        return action, ubc_scores

    
    def __bias_point(self, z):
        mu, _ = self.current_prediction()
        biased_z = self.alpha * mu + (1-self.alpha) * z
        return biased_z


    def reset_params(self, init_gp, alpha, beta, data=None):
        self.gp = cpy(init_gp)
        self.alpha = alpha.numpy()  # belief updating param
        self.beta = beta.numpy()    # UBC search param
        self.xy_queries = []
        self.z_queries = []
        biased_z = []
        if data:
            self.xy_p, self.z_p = data
        for i in range(len(self.z_p)):
            self.cur = self.xy_p[i]
            biased_mu = self.__bias_point(self.z_p[i])
            biased_z.append(biased_mu)
            self.gp.fit(self.xy_p[:i+1], biased_z)

    
    def current_prediction(self, points=None, cur=True):
        if points is None:
            if cur:
                mu, Cov = self.gp.predict(np.array(self.cur).reshape(1,-1), return_cov=True)
            else:
                mu, Cov = self.gp.predict(self.all_points, return_cov=True)
        else:
            mu, Cov = self.gp.predict(points, return_cov=True)
        return mu, Cov




