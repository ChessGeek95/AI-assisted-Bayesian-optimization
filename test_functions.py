from wandb import agent
from trial import Trial
from utils import omnisci_baseline, plot_all_trials, plot_compare, compare_trials, random_baseline
from copy import deepcopy as cpy

import numpy as np
from utils import PATH


n_trials = 50
expr = "exp_7"

init_gp_list = []
user_data_list = []
agent_data_list = []
function_list = []
trials_greedy = []

for i in range(n_trials):
    trl_greedy = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(i)+"_GreedyAI.pkl")
    trials_greedy.append(trl_greedy)
    init_gp_list.append(trl_greedy.init_gp)
    user_data_list.append(trl_greedy.user_data)
    agent_data_list.append(trl_greedy.agent_data)
    function_list.append(trl_greedy.function_df)

n_arms = trl_greedy.n_arms
n_iters = trl_greedy.n_iters

scores_greedy = np.array([trials_greedy[i].scores for i in range(n_trials)])
scores_random = random_baseline(function_list, n_arms, n_iters)
scores_omnisci = omnisci_baseline(function_list, user_data_list, agent_data_list, init_gp_list, n_arms, n_iters)

print("plotting the comaprison ...")
#plot_compare([scores_random, scores_omnisci, scores_greedy], \
#                 ["Random", "CentUCB", "Greedy"], "plot_"+expr)

#print("comparing the trials ...")
#compare_trials(scores_planning, scores_greedy, "res_"+expr)

print("plotting all trials ...")
#plot_all_trials(trials_planning, trials_greedy, expr)

random_avg = np.mean(scores_random, axis=0)
omnisci_avg = np.mean(scores_omnisci, axis=0)
greedy_avg = np.mean(scores_greedy, axis=0)

print("OmniSci :", np.round(omnisci_avg,1))
print("Greedy  :", np.round(greedy_avg,1))
print("Random  :", np.round(random_avg,1))

print("-"*60)
print("ratio Greedy  :", np.round((greedy_avg-random_avg)/(omnisci_avg-random_avg)*100,2))
