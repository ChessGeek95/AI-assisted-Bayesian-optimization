from itertools import chain #To unlist lists of lists

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

import scipy
import scipy.stats
from scipy.special import ndtr as std_normal_cdf

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from scipy.optimize import minimize


from trial import Trial
from utils import PATH

import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 


def std_normal_pdf(x):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*np.power(x,2))

def var2_normal_pdf(x):
    return (1/(np.sqrt(4*np.pi)))*np.exp(-0.25*np.power(x,2))




class UsermodelLikelihood:

    ''' Code snippets from https://github.com/AaltoPML/PPBO'''

    def __init__(self,m):
        self.m = m #number of actions the user can take minus one action
        #self.N = None #number user feedbacks times hypothetical actions, i.e. length of score vector
        self.n_gausshermite_sample_points = 40

    def is_pseudobs(self,i):
        if i==0:
            return True
        else:
            return i % (self.m+1) == 0

    def update_indices_bookeeping(self,N):
        self.N = N
        self.obs_indices = [i for i in range(0,N) if self.is_pseudobs(i)]
        self.pseudobs_indices = [i for i in range(0,N) if not self.is_pseudobs(i)]
        self.latest_obs_indices =sum([[i,]*(self.m) for i in self.obs_indices],[])


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
        Delta = f[i+1:i+m+1]
        Delta = (Delta - f[i])/sigma
        sum_=0
        if order_of_derivative==0:
            for j in range(0,m):
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
        likelihood = - np.sum(sumPhi)  #old: - np.sum(sumPhi)/self.m
        return likelihood

    

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
        ucb_scores[i*y_arms:(i+1)*y_arms] = mu.reshape(-1,) + beta * std
        mu = gp_model.predict([xy_quers[i]])
        biased_z = alpha * mu + (1-alpha) * z_quers[i]
        z_qs.append(biased_z)
        xy = np.append(xy_p, xy_quers[:i+1]).reshape(-1,2)
        z = np.append(z_p, z_qs)
        gp_model.fit(xy, z)
    return ucb_scores





#===========================================================

sigma = 0.001

expr = "exp_1"
trial_num = 0
iters_num = 7


#trial_1 = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_PlanningAI.pkl")
trial_1 = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_GreedyAI.pkl")
#print(trial_1.n_arms)
#print(trial_1.n_iters)

n_part = 4
params = [[(i/(10),j/(10)) for i in range(1, n_part+1)] for j in range(1, n_part+1)]

x_arms, y_arms = trial_1.n_arms
#alpha, beta = trial_1.user_params
print("True params: ",trial_1.user_params)
#alpha, beta = (0.1, 0.8)
xy_queries = trial_1.xy_queries[:iters_num]
z_queries = trial_1.z_queries[:iters_num]
user_xy = trial_1.user_data
user_z = trial_1.user_data_z
#print("num init points: ", len(user_xy))



def log_likelihood(theta,data):
    alpha, beta = theta
    xy_q, z_q, user_xy, user_z = data[0]
    ucb_scores = comp_ucb_scores(alpha, beta, xy_q, z_q, user_xy, user_z)
    likclass = UsermodelLikelihood(y_arms-1)
    likclass.update_indices_bookeeping(len(ucb_scores))
    ll = likclass.likelihood(ucb_scores, sigma)
    return ll


data = [(xy_queries, z_queries, user_xy, user_z)]
"""
#ML-estimate
theta = (0.2, 0.2)
nll = lambda *args: -log_likelihood(*args)
sol = minimize(nll,[0.3,0.3],args=(data), bounds=((0,1), (0,1)))
#print(sol)
"""


def log_prior(theta):
    alpha, beta = theta
    if 0 <= alpha <= 1 and 0 <= beta <=1:
        return 0.0
    return -np.inf

def log_posterior(theta):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data)






''' --- Laplace Approximation (preferred posterior inference method) --- '''

def laplace_approx(theta, theta_map, H):
    detH =  np.linalg.det(H)
    constant = np.sqrt(detH)/(2*np.pi)**(2.0/2.0)
    density = np.exp(-0.5 * (theta-theta_map).dot(H).dot(theta-theta_map))
    return constant * density

#theta_MAP and Hessian_MAP
theta_initial = np.array([0.3, 0.3]) # initial guess
solution = scipy.optimize.minimize(lambda theta: -log_posterior(theta), theta_initial, method='BFGS', options={'gtol': 1e-08})
theta_map = solution.x
covariance_matrix = solution.hess_inv #i.e. negative of log posterior at the map-estimate
#print("Cov:", covariance_matrix)
hessian = np.linalg.inv(solution.hess_inv)

#Sample from the posterior
def sample():
    draw = np.array([-1, -1])
    while np.any(draw < 0) or np.any(draw > 1):
        draw = np.random.multivariate_normal(theta_map, covariance_matrix, check_valid='warn')
    return draw

#"""
print("="*5, " samples ", "="*5)
samples = []
for i in range(2000):
    s = sample()
    samples.append(s)
    if i < 10:
        print(np.round(s, 3))

print("mean:", np.round(np.mean(samples, axis=0), 3))
print("std: ", np.round(np.std(samples, axis=0), 3))
#"""


#Plot the joint density
side = np.linspace(0,1,400)
X,Y = np.meshgrid(side,side)
Z = np.vectorize(lambda alpha,beta: laplace_approx([alpha,beta], theta_map, hessian))(X,Y)
plt.figure()
F, A1 = plt.subplots(ncols=1,figsize=(15,5))
xxm = np.ma.masked_less(Z,0.01)
cmap1 = cm.get_cmap("jet",lut=10)
cmap1.set_bad("k")
A1.set_title("Densityplot of the posterior p(alpha,beta|data)")
P = A1.pcolormesh(X,Y,xxm,cmap=cmap1)
plt.colorbar(P,ax=A1)
plt.plot(0.5,0.5,'*',color='black')
plt.annotate("true (alpha,beta)", (0.5,0.5))
plt.xlabel("alpha")
plt.ylabel("beta")
plt.savefig(PATH+'densityplot.png')
