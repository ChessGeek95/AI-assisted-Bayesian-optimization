from wandb import agent
from trial import Trial
from utils import omnisci_baseline, plot_all_trials, plot_compare, compare_trials, random_baseline
from copy import deepcopy as cpy

import numpy as np
from utils import PATH


n_trials = 18
expr = "exp_24"

init_gp_list = []
user_data_list = []
agent_data_list = []
function_list = []
trials_planning = []
trials_greedy = []

for i in range(n_trials):
    trl_planning = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(i)+"_PlanningAI.pkl")
    trials_planning.append(trl_planning)
    trl_greedy = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(i)+"_GreedyAI.pkl")
    trials_greedy.append(trl_greedy)
    init_gp_list.append(trl_greedy.init_gp)
    user_data_list.append(trl_greedy.user_data)
    agent_data_list.append(trl_greedy.agent_data)
    function_list.append(trl_greedy.function_df)

n_arms = trl_greedy.n_arms
n_iters = trl_greedy.n_iters

scores_planning = np.array([trials_planning[i].scores for i in range(n_trials)])
scores_greedy = np.array([trials_greedy[i].scores for i in range(n_trials)])
scores_random = random_baseline(function_list, n_arms, n_iters)
scores_omnisci = omnisci_baseline(function_list, user_data_list, agent_data_list, init_gp_list, n_arms, n_iters)

print("plotting the comaprison ...")
plot_compare([scores_random, scores_omnisci, scores_planning, scores_greedy], \
                 ["Random", "CentUCB", "Planning", "Greedy"], "plot_"+expr)

print("comparing the trials ...")
#compare_trials(scores_planning, scores_greedy, "res_"+expr)

print("plotting all trials ...")
#plot_all_trials(trials_planning, trials_greedy, expr)

random_avg = np.mean(scores_random, axis=0)
omnisci_avg = np.mean(scores_omnisci, axis=0)
planning_avg = np.mean(scores_planning, axis=0)
greedy_avg = np.mean(scores_greedy, axis=0)

print("OmniSci :", np.round(omnisci_avg,1))
print("Planning:", np.round(planning_avg,1))
print("Greedy  :", np.round(greedy_avg,1))
print("Random  :", np.round(random_avg,1))

print("-"*60)
print("ratio Planning:", np.round((planning_avg-random_avg)/(omnisci_avg-random_avg)*100,2))
print("ratio Greedy  :", np.round((greedy_avg-random_avg)/(omnisci_avg-random_avg)*100,2))

print("-"*60)
win_rates = np.zeros_like(scores_planning)
win_rates[scores_planning > scores_greedy] = 1
print("win rate:", np.round(np.mean(100*win_rates, axis=0),1))

print("#"*30, "  Wining  ", "#"*30)
print("planning:", np.round(np.mean(scores_planning[win_rates[:,-1]==1], axis=0),1))
print("greedy  :", np.round(np.mean(scores_greedy[win_rates[:,-1]==1], axis=0),1))


print("#"*30, "  Losing  ", "#"*30)
print("planning:", np.round(np.mean(scores_planning[win_rates[:,-1]==0], axis=0),1))
print("greedy  :", np.round(np.mean(scores_greedy[win_rates[:,-1]==0], axis=0),1))