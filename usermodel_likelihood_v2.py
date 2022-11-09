from copy import deepcopy
import numpy as np
from itertools import chain #To unlist lists of lists

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import scipy
import scipy.stats
from scipy.special import ndtr as std_normal_cdf
from wandb import agent #fast numerical integration for standard normal cdf

from trial import Trial
from utils import PATH

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def std_normal_pdf(x):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*np.power(x,2))

def var2_normal_pdf(x):
    return (1/(np.sqrt(4*np.pi)))*np.exp(-0.25*np.power(x,2))




class UsermodelLikelihood:

    def __init__(self,m):
        self.m = m #number of actions the user can take, i.e. #grid-y-points
        #self.N = None #number user feedbacks times hypothetical actions, i.e. length of score vector
        self.n_gausshermite_sample_points = 40

    def is_pseudobs(self,i):
        return i % self.m == 0

    def update_indices_bookeeping(self,N):
        self.N = N
        self.obs_indices = [i for i in range(0,N) if self.is_pseudobs(i)]
        self.pseudobs_indices = [i for i in range(0,N) if not self.is_pseudobs(i)]
        #print(self.obs_indices)
        #self.latest_obs_indices = [np.max([ind for ind in self.obs_indices if ind < i]) if self.is_pseudobs(i) else i for i in range(0,N)]


    ''' Auxiliary functions for computing transormations/vectroizations of Phi '''
    def sum_Phi(self,i,order_of_derivative,f,sigma,sample_points=None, weights=None):
        '''
        f = ucb scores
        Auxiliary function that computes summation of Phi-function values
        i = index of x observation, i.e. some element of list 'obs_indices'
        order_of_derivative = how many this the c.d.f of the standard normal (Phi) is differentiated?
        f = vector of f-values (i.e. UCB score values)
        '''
        m = self.m
        #Delta_i_j = (f[i+j+1]-f[i])/(sigma_)
        Delta = f[i+1:i+m]
        #print(i, ' => ', len(Delta))
        Delta = (Delta - f[i])/sigma
        sum_=0
        if order_of_derivative==0:
            for j in range(m-1):
                #sum_ = sum_ + integrate(lambda x: std_normal_cdf(Delta[j]+x)*std_normal_pdf(x), -np.inf, np.inf)[0]
                #Do integration by using Gaussian-Hermite quadrature:
                sum_ = sum_ + (1/np.sqrt(np.pi))*np.dot(weights,std_normal_cdf(Delta[j]-np.sqrt(2)*sample_points)) #Do to change of variables to get integteragl into the form int exp(x^2)*....dx. This is why np.sqrt(2)
            return(sum_)
        if order_of_derivative==1:
            for j in range(0,m):
                sum_ = sum_ + float(var2_normal_pdf(Delta[j]))
            return(sum_)
        if order_of_derivative==2:
            for j in range(0,m):
                sum_ = sum_ - 0.5*Delta[j]*float(var2_normal_pdf(Delta[j]))
            return(sum_)
        else:
            print("The derivatives of an order higher than 2 are not needed!")
            return None

    def sum_Phi_vec(self,order_of_derivative,f,sigma,over_all_indices=False):
        '''
        f = ucb scores
        Auxiliary function that create a vector of sum_Phi with elements as
        i = obs_indices[0],...,obs_indices[len(obs_indices)],   if not over_all_indices
        i = 1,...,N                                      if over_all_indices
        '''
        sample_points, weights = np.polynomial.hermite.hermgauss(self.n_gausshermite_sample_points) #for cross-correlation integral
        if not over_all_indices:
            sum_Phi_vec_ = [self.sum_Phi(i,order_of_derivative,f,sigma,sample_points, weights) for i in self.obs_indices]
            return np.array(sum_Phi_vec_)
        else:
            sum_Phi_vec_ = list(chain.from_iterable([[self.sum_Phi(i,order_of_derivative,f,sigma,sample_points, weights)]*(self.m+1) for i in self.obs_indices])) #for z indices just replicate previous x value
            return np.array(sum_Phi_vec_)

    ''' Derivatives of the functional T of the order: 0,1 '''
    ''' f = vector of UCB score values '''
    ''' len(f) = num_user_feedback x num_actions_over_y '''

    def likelihood(self,ucb_scores,sigma):
        sumPhi = self.sum_Phi_vec(0,ucb_scores,sigma)
        likelihood = - np.sum(sumPhi)/self.m
        return likelihood

    # def likelihood_grad(self,ucb_scores,sigma):
    #     N = self.N
    #     m = self.m
    #     latest_obs_indices = self.latest_obs_indices
    #     beta = np.zeros((N,1)).reshape(N,)
    #     Phi_der_vec_ = np.array([float(var2_normal_pdf((ucb_scores[j]-ucb_scores[latest_obs_indices[j]])/sigma)) for j in self.pseudobs_indices])
    #     sum_Phi_der_vec_ = self.sum_Phi_vec(1,ucb_scores,sigma)
    #     beta[self.obs_indices] = sum_Phi_der_vec_/(sigma*m)
    #     beta[self.pseudobs_indices] = -Phi_der_vec_/(sigma*m)
    #     likelihood_grad = beta.reshape(N,)
    #     return likelihood_grad
    #
    # def max_likelihood(self):
    #     #theta = (alpa,beta,gamma)
    #     min_ = 10**24
    #     trials = 5
    #     for i in range(0,trials):
    #         theta_initial = [0,0,0] #some prior distribution
    #         res = scipy.optimize.minimize(lambda theta: -self.likelihood(ucb_scores(theta[0],theta[1]),theta[2]), theta_initial,jac=lambda f: -self.ucb_scores_grad(f,self.theta))
    #         if res.fun < min_:
    #             min_ = res.fun
    #             besttheta = res.x
    #     return besttheta





''' Example how this class can be used '''

def comp_ucb_scores(alpha, beta, xy_quers, z_quers, xy_p, z_p):
    
    xy = np.array(xy_quers)
    agent_actions = xy[:,0]
    user_actions = xy[:,1]
    x = np.repeat(agent_actions, y_arms).reshape(-1, y_arms)
    y = np.array([[u]+[v for v in range(y_arms) if v!=u] for u in user_actions])

    kernel = ConstantKernel(5, constant_value_bounds="fixed") * RBF(10, length_scale_bounds="fixed")
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=2e-2)
    gp_model.fit(user_xy, user_z)

    xy_quers = np.array(xy_quers)
    n_iters = xy_quers.shape[0]
    z_qs = []
    ucb_scores = np.zeros(n_iters * y_arms)
    for i in range(n_iters):
        points = np.vstack((x[i].flatten(), y[i].flatten())).T
        mu, std = gp_model.predict(points, return_std=True)
        ucb_scores[i*y_arms:(i+1)*y_arms] = mu.reshape(-1,) + beta * np.sqrt(np.abs(std))

        mu = gp_model.predict([xy_quers[i]])
        biased_z = alpha * mu + (1-alpha) * z_quers[i]
        z_qs.append(biased_z)
        xy = np.append(xy_p, xy_quers[:i+1]).reshape(-1,2)
        z = np.append(z_p, z_qs)
        gp_model.fit(xy, z)
        

    return ucb_scores





"""
def comp_ucb_scores(alpha, beta, xy_quers, z_quers):
    # BO_data = {(x_1, y*_1, f(x_1, y*_1)), (x_2, y*_2, f(x_2, y*_2)), (x_3, y*_3, f(x_3, y*_3))}
    # outputs ucb_scores = { ucb(x_1,y*_1), ucb(x_1,y^(1)),..., ucb(x_1,y^(m)),
    #                        ucb(x_2,y*_2), ucb(x_2,y^(1)),..., ucb(x_2,y^(m)),
    #                        ucb(x_3,y*_3), ucb(x_3,y^(1)),..., ucb(x_3,y^(m)) }

    # outputs ucb_scores = { ucb(x_1,y*_1|alpa,beta,BO_data,userfeedback_data), ucb(x_1,y^(1)|alpa,beta,BO_data,userfeedback_data),...
    # ...,ucb(x_1,y^(m)|alpa,beta,BO_data,userfeedback_data),ucb(x_2,y*_2|alpa,beta,BO_data,userfeedback_data),... )
    # len(ucb_scores) = num_user_feedbacks x num_possible_user_actions

    kernel = ConstantKernel(5, constant_value_bounds="fixed") * RBF(10, length_scale_bounds="fixed")
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3)

    xy_quers = np.array(xy_quers)
    n_iters = xy_quers.shape[0]
    z = []
    for i in range(n_iters):
        mu = gp_model.predict([xy_quers[i]])
        biased_z = alpha * mu + (1-alpha) * z_quers[i]
        z.append(biased_z)
        gp_model.fit(xy_quers[:i+1], z)
    
    xy = np.array(xy_quers)
    agent_actions = xy[:,0]
    user_actions = xy[:,1]
    x = np.repeat(agent_actions, y_arms).reshape(-1, y_arms)
    y = np.array([[u]+[v for v in range(y_arms) if v!=u] for u in user_actions])
    points = np.vstack((x.flatten(), y.flatten())).T
    mu, std = gp_model.predict(points, return_std=True)
    ucb_scores = mu.reshape(-1,) + beta * np.sqrt(np.abs(std))

    return ucb_scores
"""



#===========================================================

sigma = 0.001


expr = "exp_27"
trial_num = 2
iters_num = 5


#trial_1 = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_PlanningAI.pkl")
trial_1 = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_GreedyAI.pkl")
#print(trial_1.n_arms)
#print(trial_1.n_iters)

n_part = 4
params = [[(i/(10),j/(10)) for i in range(1, n_part+1)] for j in range(1, n_part+1)]
#params = [[(0.2, 0.2)]]

x_arms, y_arms = trial_1.n_arms
#alpha, beta = trial_1.user_params
print("True params: ",trial_1.user_params)
#alpha, beta = (0.1, 0.8)
xy_queries = trial_1.xy_queries[:iters_num]
z_queries = trial_1.z_queries[:iters_num]
user_xy = trial_1.user_data
user_z = trial_1.user_data_z
print("num init points: ", len(user_xy))

"""
alpha, beta = trial_1.user_params
xy_q = deepcopy(xy_queries)
z_q = deepcopy(z_queries)
ucb_scores = comp_ucb_scores(alpha, beta, xy_q, z_q)

"""

#===========================================================
ll_list = np.zeros_like(params)
for ii in range(ll_list.shape[0]):
    for jj in range(ll_list.shape[1]):
        alpha, beta = params[ii][jj]
        xy_q = deepcopy(xy_queries)
        z_q = deepcopy(z_queries)
        ucb_scores = comp_ucb_scores(alpha, beta, xy_q, z_q, user_xy, user_z)
        #print(ucb_scores.shape)
        #print(ucb_scores[:y_arms+1])
        
        #######################################################
        likclass = UsermodelLikelihood(y_arms)
        likclass.update_indices_bookeeping(len(ucb_scores))
        ll = likclass.likelihood(ucb_scores, sigma)
        ll_list[ii,jj] = ll
        ######################################################


print("="*40)
for i in range(n_part):
    for j in range(n_part):
        #print(np.round(params[j][i], 2), " ==> ", np.round(ll_list[j,i][0],3))
        print(np.round(params[i][j], 2), " ==> ", np.round(ll_list[i,j][0],3))
    
#"""