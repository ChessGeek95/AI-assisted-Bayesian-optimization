import elfi
import scipy
import numpy as np


class ABC:
    def __init__(self, obs_data, generative_model, y_arms):
        self.y_arms = y_arms
        self.obs_data = obs_data
        self.generative_model = generative_model

    
    def new_model(self):
        self.my_model = elfi.ElfiModel()
        alpha = elfi.Prior(scipy.stats.uniform, 0, 1, model=self.my_model)
        beta = elfi.Prior(scipy.stats.uniform, 0, 1, model=self.my_model)
        self.simulator = elfi.Simulator(self.generative_model, alpha, beta, observed=self.obs_data, model=self.my_model)
        self.result_smc = None
        self.means = (None, None)
        
        def SummaryStat(sim):
            return sim
        
        
        def my_distance(XA, YA):
            dist = np.abs(XA - YA)/self.y_arms
            dist = np.mean(dist, axis=1)
            return dist
        
        """
        def my_distance(XA, YA):
            dist = np.sum((XA - YA) ** 2, axis=1)
            dist[dist!=0] = 1
            return dist
        """

        self.suf_stat = elfi.Summary(SummaryStat, self.simulator, model=self.my_model)
        self.distance = elfi.Distance(my_distance, self.suf_stat, model=self.my_model)

    
    def fit(self):
        if len(self.obs_data) == 0:
            alpha_samples = np.random.random(2000)
            beta_samples = np.random.random(2000)
            self.means = (0.5, 0.5)
            return alpha_samples, beta_samples, self.means
        #return self.seq_monte_carlo()
        return self.rejection_sampling()
        

    def seq_monte_carlo(self):
        smc = elfi.SMC(self.distance, batch_size=1)
        N = 100
        schedule = [0.95, 0.75, 0.3]
        self.result = smc.sample(N, schedule)
        alpha_samples = self.result.populations[-1].samples['alpha']
        beta_samples = self.result.populations[-1].samples['beta']
        self.means = (alpha_samples, beta_samples)
        self.result.summary()
        return alpha_samples, beta_samples, self.means

    
    def rejection_sampling(self):
        rej = elfi.Rejection(self.distance, batch_size=1)
        N = 100
        self.result = rej.sample(N, quantile=0.01)
        alpha_samples = self.result.samples['alpha']
        beta_samples = self.result.samples['beta']
        self.means = (alpha_samples, beta_samples)
        self.result.summary()
        return alpha_samples, beta_samples, self.means
    
    
    def reset_model(self):
        del self.my_model
        


