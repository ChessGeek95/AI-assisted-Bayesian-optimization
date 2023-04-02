import pathlib
import numpy as np
import matplotlib
import os
import argparse, sys

from experiment_settings import ExperimentSettings
from trial import Trial
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from copy import deepcopy
import datetime
from joblib import Parallel, delayed
import time

from interface import Interface
from users.simulated_user import GreedyUser
from agents.agents import BaseAgent, EpsGreedyAgent, StrategicAgent

from utils import PATH, RANDOM_SEED, create_readme
import warnings
from sklearn.exceptions import ConvergenceWarning

#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


sys.path.append(PATH)


def simulate(interface, user, agent, experiment_settings, trials_range, n_iters, expr, known_user=False, name=None, n_jobs=1):
    np.random.seed(RANDOM_SEED)
    start = time.time()
    if n_jobs > 1:
        Parallel(n_jobs=n_jobs)(delayed(simulate_trial)(episode, interface, user, agent,
                                         experiment_settings, n_iters, expr, name, known_user) for episode in range(*trials_range))
    else:
        for episode in range(*trials_range):
            simulate_trial(episode, interface, user, agent, experiment_settings, n_iters, expr, name, known_user)
    end = time.time()
    #====================================================================
    with open(EXP_PATH+"run.log", "a") as f:
        f.write(expr + "  trial_rng:" + str(trl_range) +
                "  user_pars:" + str((user.alpha, user.beta))+ " user_idx:"+ str(user_idx)+
                "   agent:" + name + "  time:" + '{:.1f} s'.format(end-start) + "\n")
    return 0


def simulate_trial(episode, interface, user, agent, experiment_settings, n_iters, expr, name, known_user):
    trial_path = EXP_PATH+expr+"/tiral_"+str(experiment_settings.id)+"_"+str(episode)+"_"+name+".pkl"
    """
    if os.path.exists(EXP_PATH):
        print("*"*30)
        print(trial_path)
        print("*"*30)
        return
    """
    
    trial = Trial()
    trial.set_values(user, agent, interface, experiment_settings, n_iters, episode)
    #func_arg = experiment_settings.function_args[episode]
    start_point = experiment_settings.starting_points[episode]

    user_data, agent_data = interface.reset(expr_settings.function_list[episode],
                                            expr_settings.user_data_list[episode],
                                            expr_settings.agent_data_list[episode],
                                            start_point)
    user.reset(user_data)
    if known_user:
        agent.reset(agent_data, user_data) # uses perfect user knowledge, estimates params
    else:
        agent.reset(agent_data)
    
    trial.add_prior_data(agent_data, user_data)
    #trial.add_belief(user.current_prediction(cur=False), 
    #                agent.current_prediction(cur=False))
        
    trl_scores = np.zeros((n_iters,))
    for t in range(n_iters):
        x_t = agent.take_action()
        y_t = user.take_action(agent_action=x_t)
        observation = interface.step(x_t, y_t)
        agent.update(observation, y_t)
        user.update(observation)

        score = interface.get_score()
        trl_scores[t] = score[1]

        user_belief = user.current_prediction(cur=False)
        agent_belief = agent.current_prediction(cur=False)
        #trial.add_belief(user_belief, agent_belief)

    trial.add_queries(interface.xy_queries, interface.z_queries, trl_scores)
    trial.save(path=trial_path)






if __name__ == "__main__":


    #====================================================================
    
    parser=argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", help="Which seed?")
    parser.add_argument("-j", "--n_jobs", help="How many CPUs?")
    parser.add_argument("-nt", "--n_trials", help="How many trials?")
    parser.add_argument("-ni", "--n_iters", help="How many iterations?")
    parser.add_argument("-na", "--n_arms", nargs="+",  help="Space dimension?")
    parser.add_argument("-eid", "--expr_id", help="which experiment?")
    parser.add_argument("-u", "--user_idx", help="which user?")
    parser.add_argument("-rng", "--trl_range", nargs="+", help="int or tuple, which trials?")
    
    args=parser.parse_args()

    
    RANDOM_SEED  = eval(args.seed)
    N_JOBS = eval(args.n_jobs)
    N_TRIALS = eval(args.n_trials)
    N_ITERS = eval(args.n_iters)
    N_ARMS = [eval(r) if isinstance(r, str) else r for r in args.n_arms]
    
    EXPERIMENT_ID = eval(args.expr_id)
    user_idx = eval(args.user_idx)
    trl_range = [eval(r) if isinstance(r, str) else r for r in args.trl_range]


    if len(trl_range)==1:
        trl_range = [trl_range[0], trl_range[0]+1]
    else:
        trl_range[1] += 1
    if trl_range[1] > N_TRIALS:
        trl_range[1] = N_TRIALS
    
    #====================================================================
    
    #=== init params
    
    ALPHA_SET = [0.1, 0.6]
    BETA_SET = [0.2, 0.7]
    BETA_UCB = 0.05
    THETA_USER_SET = [(a, b) for a in ALPHA_SET for b in BETA_SET]
    N_ARMS = (50, 50)
    GENERATE_NEW_EXP = False
    EXP_PATH = PATH+"trials/expriment_"+str(EXPERIMENT_ID)+"/"

    experiment_sets = ["EXP_"+str(v) for v in range(len(THETA_USER_SET))]

    np.random.seed(RANDOM_SEED)
    #====================================================================
    
    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH)
        os.makedirs(EXP_PATH + "problems/")
        os.makedirs(EXP_PATH + "results/")
        for ex in experiment_sets:
            os.makedirs(EXP_PATH + ex + "/")
    
    #====================================================================
    expr_settings = ExperimentSettings()
    if (not os.listdir(EXP_PATH+"problems/")) or GENERATE_NEW_EXP:
        print("generating new set of problems...")
        expr_settings.set_values(N_ARMS, N_TRIALS, id=EXPERIMENT_ID)
        expr_settings.save(path=EXP_PATH+"problems/problems.pkl")
    else:
        expr_settings.load(path=EXP_PATH+"problems/problems.pkl")


    #====================================================================
    exp_info = [(experiment_sets[i], THETA_USER_SET[i]) for i in range(len(experiment_sets))]
    create_readme(EXP_PATH, ["RANDOM_SEED", RANDOM_SEED],
                            ["N_TRIALS",    N_TRIALS],
                            ["N_ITERS",     N_ITERS],
                            ["ALPHA_SET",   ALPHA_SET],
                            ["BETA_SET",    BETA_SET],
                            ["BETA_UCB",    BETA_UCB],
                            ["THETA_U",     THETA_USER_SET],
                            ["N_ARMS",      N_ARMS],
                            *exp_info)
    #====================================================================
    
    #"""
    #=== init the environment
    interface = Interface(N_ARMS)
    
    #=== init the synthetic user
    kernel = ConstantKernel(5, constant_value_bounds="fixed") * RBF(10, length_scale_bounds="fixed")
    #kernel = ConstantKernel(5, constant_value_bounds="fixed") * RBF(10, length_scale_bounds=(5, 20))
    #gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=2e-2)
    users = []
    for user_params in THETA_USER_SET:
        gp_model = GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10, alpha=2e-2)
        user = GreedyUser(gp_model, N_ARMS, interface.get_cur(), user_params)
        users.append(user)

    #=== init the AI agents
    gp_model_random = GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10, alpha=2e-2)
    random_agent = BaseAgent(gp_model_random, N_ARMS, interface.get_cur(), max_iters=N_ITERS)
    
    gp_model_greedy = GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10, alpha=2e-2)
    greedy_agent = EpsGreedyAgent(gp_model_greedy, N_ARMS, interface.get_cur(), beta=BETA_UCB, max_iters=N_ITERS)

    #the planning agent with known user knowledge
    gp_model_omni_stratgic = GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10, alpha=2e-2)
    omni_strategic_agent = StrategicAgent(gp_model_omni_stratgic, N_ARMS, interface.get_cur(), max_iters=N_ITERS)
    
    gp_model_stratgic = GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10, alpha=2e-2)
    strategic_agent = StrategicAgent(gp_model_stratgic, N_ARMS, interface.get_cur(), max_iters=N_ITERS)

    gp_model_omni_stratgic_2 = GaussianProcessRegressor(kernel=deepcopy(kernel), n_restarts_optimizer=10, alpha=2e-2)
    omni_strategic_agent_2 = StrategicAgent(gp_model_omni_stratgic_2, N_ARMS, interface.get_cur(), max_iters=N_ITERS, user_params=THETA_USER_SET[user_idx])
    #====================================================================
    #=== run the experiment
    #"""
    
    #"""
    
    simulate(interface, users[user_idx], random_agent, expr_settings, trl_range, N_ITERS,
                expr=str(experiment_sets[user_idx]), known_user=False, name="RandomAI")

    
    
    simulate(interface, users[user_idx], greedy_agent, expr_settings, trl_range, N_ITERS,
                expr=str(experiment_sets[user_idx]), known_user=False, name="GreedyAI_2")

    #"""

    simulate(interface, users[user_idx], strategic_agent, expr_settings, trl_range, N_ITERS, 
                expr=str(experiment_sets[user_idx]), known_user=False, name="PlanningAI", n_jobs=N_JOBS)
    
    """
    simulate(interface, users[user_idx], omni_strategic_agent, expr_settings, trl_range, N_ITERS, 
                expr=str(experiment_sets[user_idx]), known_user=True, name="OmniPlanningAI", n_jobs=N_JOBS)
    
    simulate(interface, users[user_idx], omni_strategic_agent_2, expr_settings, trl_range, N_ITERS, 
                expr=str(experiment_sets[user_idx]), known_user=True, name="OmniPlanningAI_2", n_jobs=N_JOBS)

    #"""