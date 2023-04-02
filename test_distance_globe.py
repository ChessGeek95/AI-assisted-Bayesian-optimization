from trial import Trial
from utils import omnisci_baseline, plot_all_trials, plot_compare, compare_trials, random_baseline
from copy import deepcopy as cpy

import numpy as np
from utils import PATH


n_trials = 50
expr = "exp_"


dist_planning = []
dist_greedy = []
dist_omnisci = []
dist_random = []

init_gp_list = []
user_data_list = []
agent_data_list = []
function_list = []


def get_distance(g_idx, xy_queries):
    idx_y = g_idx//80
    idx_x = g_idx%80
    dists = []
    for x,y in xy_queries:
        d = ((x-idx_x)**2 + (y-idx_y)**2)**0.5
        dists.append(d)
    best_idx = np.argmin(dists)
    best_xy = xy_queries[best_idx]
    best_dist = dists[best_idx]
    return best_dist#, best_xy, dists




for i_exp in range(3,8):
    for i in range(n_trials):
        trl_planning = Trial(PATH + "trials/"+expr+str(i_exp)+"/tiral_100_"+str(i)+"_PlanningAI.pkl")
        trl_greedy = Trial(PATH + "trials/"+expr+str(i_exp)+"/tiral_100_"+str(i)+"_GreedyAI.pkl")

        g_idx = np.argmax(trl_planning.function_df)
        dist_p = []
        for k in range(trl_planning.n_iters):
            dp = get_distance(g_idx, trl_planning.xy_queries[:k+1])
            dist_p.append(dp)
        dist_g = []
        for k in range(trl_greedy.n_iters):
            dg = get_distance(g_idx, trl_greedy.xy_queries[:k+1])
            dist_g.append(dg)
        
        dist_planning.append(dist_p)
        dist_greedy.append(dist_g)
        init_gp_list.append(trl_greedy.init_gp)
        user_data_list.append(trl_greedy.user_data)
        agent_data_list.append(trl_greedy.agent_data)
        function_list.append(trl_greedy.function_df)
        


print("n_trials:", len(dist_planning))
##################
params_u = trl_planning.user_params
n_iters = trl_greedy.n_iters
n_arms = trl_planning.n_arms
max_dist = np.sqrt(n_arms[0]**2 + n_arms[1]**2)

##################

#_, xy_random = random_baseline(function_list, n_arms, n_iters)
_, xy_omnisci = omnisci_baseline(function_list, user_data_list, agent_data_list, init_gp_list, n_arms, n_iters)

for i_exp in range(3,8):
    for i in range(n_trials):
        ii = (i_exp-3)*n_trials+i

        g_idx = np.argmax(function_list[ii])
        dist_ucb = []
        for k in range(n_iters):
            d_ = get_distance(g_idx, xy_omnisci[ii][:k+1])
            dist_ucb.append(d_)
        
        dist_omnisci.append(dist_ucb)



dist_planning = np.array(dist_planning)/(max_dist)
dist_greedy = np.array(dist_greedy)/(max_dist)
#dist_omnisci = np.array(dist_omnisci)/(max_dist)
#dist_random = np.array(dist_random)/(max_dist)

print("plotting the comaprison ...")
#plot_compare([scores_random, scores_omnisci, scores_greedy, scores_planning], \
#                 ["Random", "UCB", "Greedy AI", "Planning AI"], params_u, "plot_"+expr+"all")
plot_compare([dist_greedy, dist_planning], \
                 ["Greedy AI", "Planning AI"], params_u, "plot_distance_"+expr+"all")

print("comparing the trials ...")
#compare_trials(scores_planning, scores_greedy, "res_"+expr)

print("plotting all trials ...")
#plot_all_trials(trials_planning, trials_greedy, expr)

#random_avg = np.mean(scores_random, axis=0)
#omnisci_avg = np.mean(dist_omnisci, axis=0)
planning_avg = np.mean(dist_planning, axis=0)
greedy_avg = np.mean(dist_greedy, axis=0)

print()
#print("UCB :", np.round(omnisci_avg,3))
print("Planning:", np.round(planning_avg,3))
print("Greedy  :", np.round(greedy_avg,3))
#print("Random  :", np.round(random_avg,1))


#print("-"*60)
#print("ratio Planning:", np.round((planning_avg-random_avg)/(omnisci_avg-random_avg)*100,2))
#print("ratio Greedy  :", np.round((greedy_avg-random_avg)/(omnisci_avg-random_avg)*100,2))

print("-"*60)
win_rates = np.zeros_like(dist_planning)
win_rates[dist_planning < dist_greedy] = 1
print("win rate:", np.round(np.mean(100*win_rates, axis=0),2))



print("#"*30, "  Wining  ", "#"*30)
print("planning:", np.round(np.mean(dist_planning[win_rates[:,-1]==1], axis=0),3))
print("greedy  :", np.round(np.mean(dist_greedy[win_rates[:,-1]==1], axis=0),3))


print("#"*30, "  Losing  ", "#"*30)
print("planning:", np.round(np.mean(dist_planning[win_rates[:,-1]==0], axis=0),3))
print("greedy  :", np.round(np.mean(dist_greedy[win_rates[:,-1]==0], axis=0),3))

#"""