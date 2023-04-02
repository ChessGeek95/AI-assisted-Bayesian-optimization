from trial import Trial
from utils import omnisci_baseline, plot_all_trials, plot_compare, compare_trials, random_baseline
from copy import deepcopy as cpy
from experiment_settings import ExperimentSettings

import numpy as np
from utils import PATH, get_scores

from joblib import Parallel, delayed
import warnings, os
warnings.filterwarnings("ignore")


EXPERIMENT_ID = 22
EXPERIMENT_ID_2 = 21
expr_path = PATH+"trials/expriment_"+str(EXPERIMENT_ID)+"/"

expr_settings = ExperimentSettings()
expr_settings.load(path=expr_path+"problems/problems.pkl")



n_iters = 20
n_arms = expr_settings.n_arms
n_trials = expr_settings.n_trials
n_data_types = 7


### ==========================================  Reading the data  ===============================================

users_set = np.arange(6)
alpha_set = [0.0, 0.2, 0.6]
beta_set = [0.2, 0.7]
param_set = [(a, b) for a in alpha_set for b in beta_set]


trials_ranges = []
batch_size = n_trials//n_data_types
for i in range(n_data_types):
    trials_ranges.append((i*batch_size, (i+1)*batch_size))

data_decrp = ["AI: Glob   User: Loc", "AI: Loc   User: Globe", "Both: Glob", "None: Globe",
                "AI: Glob   User: None", "AI: Loc   User: None", "AI: None   User: None",
                "AI: Glob   User: Combined", "AI: G|L    User: Glob", "AI: G|L    User: None",
                "AI: Combined   User: None", "All combined"]




for dt_dsc in data_decrp:
    if not os.path.exists(expr_path+"results/"+dt_dsc):
        os.makedirs(expr_path+"results/"+dt_dsc)


init_gp_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
user_data_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
agent_data_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
function_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]




for i_exp in users_set:
    for i_rng, trl_rng in enumerate(trials_ranges):
        for i_trl in range(*trl_rng):
            
            trl_greed = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_GreedyAI_2.pkl")
            

            init_gp_list[i_exp][i_rng].append(trl_greed.init_gp)
            user_data_list[i_exp][i_rng].append(trl_greed.user_data)
            agent_data_list[i_exp][i_rng].append(trl_greed.agent_data)
            function_list[i_exp][i_rng].append(trl_greed.function_df)
            
### ==================================================================================================




def plot_performance(my_users, my_rng, i_rng, score_idx=0, indices=np.arange(8), draw_trajectories=False):
    params = [param_set[i] for i in my_users]
    caption = r'params ($\alpha$, $\beta$): ' + str(params) + '\n'
    score_labels = ["optimization score", "Simple regret", "Normalized distance"]

    gp_list = [gp for i in my_users for j in my_rng for gp in init_gp_list[i][j]]
    usr_data_list = [dt for i in my_users for j in my_rng for dt in user_data_list[i][j]]
    agt_data_list = [dt for i in my_users for j in my_rng for dt in agent_data_list[i][j]]
    func_list = [func for i in my_users for j in my_rng for func in function_list[i][j]]

    all_scores = []
    for top_k in range(1, 16, 2):
        scores_single_omnisci_agent = np.array(omnisci_baseline(func_list, None, agt_data_list, gp_list, n_arms, n_iters, top_k))
        all_scores.append(scores_single_omnisci_agent)
    

    legends = ["UCB top-"+str(top_k) for top_k in range(1,16,2)]
    
    plot_compare(all_scores, legends,
                    expr_path+"results/"+score_labels[score_idx].split()[1]+"_users("+"-".join(str(x) for x in my_users)+")", 
                    indices, caption, score_labels[score_idx])







users_comb = [[0], [1], [2], [3], [4], [5],
                [0, 2, 4], [1, 3, 5],
                [0, 1], [2, 3], [4, 5],
                [0, 1, 2, 3, 4, 5]]

rng_comb = [[0], [1], [2], [3], [4], [5], [6],
            [0, 2, 4], [1, 2], [4, 5], [4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6]]

#plot_performance([0, 2], [6], i_rng=6, indices=np.arange(7), draw_trajectories=True)

#plot_performance([0, 2], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))
#plot_performance([0, 2, 4], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))
#plot_performance([1, 3, 5], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))
#plot_performance([0,1,2,3,4,5], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))

#plot_performance([0], [0], 0, indices=np.arange(4), draw_trajectories=True)

plot_performance([0], [0], i_rng=0)

"""
for usrs in users_comb:
    i_rng = 6
    rng = rng_comb[i_rng]
    plot_performance(usrs, rng, i_rng)
#"""

"""
for usrs in users_comb:
    for i_rng, rng in enumerate(rng_comb):
        #plot_performance(usrs, rng, i_rng)
        plot_performance(usrs, rng, i_rng, indices=[0,1,2,4,5,6,8])

#"""

#plot_performance([0], [0,1])


#for rng in rng_comb:
    #plot_params([4,5], rng)
#plot_params([1, 3, 5], [0, 1, 2])



#print("comparing the trials ...")
#compare_trials(scores_planning, scores_greedy, "res_"+expr)

#print("plotting all trials ...")
#plot_all_trials(trials_planning, trials_greedy, expr_path+"results/plot_compare_usr"+str(my_users)+"_trls"+str(my_rng))

