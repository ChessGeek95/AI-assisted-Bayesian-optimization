from trial import Trial
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from scipy.stats import differential_entropy

import pathlib
import numpy as np
from utils import PATH, get_scores




EXPERIMENT_ID = 143
EXPERIMENT_ID_2 = 141
expr_path = PATH+"trials/expriment_"+str(EXPERIMENT_ID)+"/"
i_exp = 1
i_trl = 2


def entropyGPmaximizer(mu,Sigma, N=50000):
    #print(mu.shape, Sigma.shape)
    fvals = np.random.multivariate_normal(mu,Sigma,N)
    argmaxsamples = np.argmax(fvals,axis=1)
    #Empirical distribution
    #print(argmaxsamples)
    #print('-'*40)
    counts = np.bincount(argmaxsamples)
    distribution = counts/N
    distribution += 0.000001
    distribution /= np.sum(distribution)
    #print(distribution)
    #Emprical entropy
    base = np.e
    entropy = -np.sum(distribution * np.log(distribution)) / np.log(base)
    return entropy



def entropyGPmaximum(mu,Cov,nmcmc=50000):
    fvals = np.random.multivariate_normal(mu,Cov,nmcmc)
    maxsamples = np.max(fvals, axis=1)
    entropy = differential_entropy(maxsamples)
    return entropy

#mu = np.array([0,0.5,0.5,1])  #GP predictive mean (over whole space/grid)
#Sigma = np.diag([2,2,2,2])  #GP predictive covariance matrix (over whole space/grid)
#entropyGPmaximizer(mu,Sigma)



def calc_entropy(trl, alpha):
    x, y = np.meshgrid(np.arange(trl.n_arms[0]), np.arange(trl.n_arms[1]))
    all_points = np.vstack((x.flatten(), y.flatten())).T #shape(x_arms*y_arms, 2)
    
    kernel = ConstantKernel(5, constant_value_bounds="fixed") * RBF(10, length_scale_bounds="fixed")
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=2e-2)
    xy_p, z_p = trl.user_data, trl.user_data_z
    gp_model.fit(xy_p, z_p)
    xy_quers = np.array(trl.xy_queries)
    z_quers = np.array(trl.z_queries)
    n_iters = xy_quers.shape[0]
    z_qs = []
    entropy = np.zeros(n_iters)
    for i in range(n_iters):
        mu_, Cov_ = gp_model.predict(all_points, return_std=False, return_cov=True)
        #print(mu.shape)
        #print(Cov.shape)
        #mu, std = gp_model.predict(points, return_std=True)
        #cov_det = np.linalg.det(Cov + 1e-10 * np.eye())
        #print("*"*10,"Cov-det", cov_det)
        #entropy[i] = 0.5 * np.log(cov_det) + 2500/2 * (1 + np.log(2*np.pi))
        

        mu = gp_model.predict([xy_quers[i]])
        biased_z = alpha * mu + (1-alpha) * z_quers[i]
        z_qs.append(biased_z)
        xy = np.append(xy_p, xy_quers[:i+1]).reshape(-1,2)
        z = np.append(z_p, z_qs)
        gp_model.fit(xy, z)

        if i == n_iters-1:
            #print(mu_[:5])
            #print(Cov_[:5,:5])
            entropy[i] = entropyGPmaximum(mu_, Cov_)
            entropy[i-1] = entropyGPmaximizer(mu_, Cov_)

        #if i>2:
            #break
    return entropy



trl = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_PlanningAI.pkl")
#trl = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_GreedyAI_2.pkl")

#print(len(trl.user_beliefs))
#print(trl.user_beliefs[-1][0].shape)
#print(trl.user_beliefs[-1][1].shape)
entropy = calc_entropy(trl, alpha=0.1)

print("Entropy_1:", entropy[-1])
print("Entropy_2:", entropy[-2])

#print(np.round(get_scores(trial)[:2],2))

#print(type(trial))
#print(trial.user_data)
#print(trial.user_data_z)

#print(trial.agent_data)
#print(trial.agent_data_z)
#print(trial.function_df[:5,:5])
#print(trial_1.scores)
#print(trial_2.scores)
#print(trial_2.user_beliefs[4][0].shape)
#print(trial_2.agent_beliefs[4][0].shape)
#print(trial.agent_beliefs[0][1].shape)
#print(trial.user_beliefs[0][1].shape)

#trial_1.plot_compare_to(trial_2, expr+"_trl_"+str(trial_num))
#trial_1.plot()

#print(trial_1.user_beliefs[1][0][:10,10])
#np.set_printoptions(suppress=True)
#print(np.round(10*get_scores(trl),2))

"""
scores = []
scores_2 = []

for i_trl in range(300,350):
    trl = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_GreedyAI_2.pkl")
    scores.append(get_scores(trl)[0])
    scores_2.append(trl.scores)

print(np.mean(scores, axis=0))
print("="*30)
print(np.mean(scores_2, axis=0))
"""