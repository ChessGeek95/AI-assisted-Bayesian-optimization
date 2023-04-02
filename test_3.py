from trial import Trial
from utils import omnisci_baseline, plot_all_trials, plot_compare, compare_trials, random_baseline
from copy import deepcopy as cpy
from experiment_settings import ExperimentSettings

import numpy as np
from utils import PATH, get_scores

from joblib import Parallel, delayed
import warnings, os
warnings.filterwarnings("ignore")


EXPERIMENT_ID = 141
EXPERIMENT_ID_2 = 141



expr_path = PATH+"trials/expriment_"+str(EXPERIMENT_ID)+"/"

expr_settings = ExperimentSettings()
expr_settings.load(path=expr_path+"problems/problems.pkl")



n_iters = 20
n_arms = expr_settings.n_arms
n_trials = expr_settings.n_trials
n_data_types = 7


### ==========================================  Reading the data  ===============================================


alpha_set = [0.1, 0.6]
beta_set = [0.2, 0.7]
param_set = [(a, b) for a in alpha_set for b in beta_set]
users_set = np.arange(len(param_set))


trials_ranges = []
batch_size = n_trials//n_data_types
for i in range(n_data_types):
    trials_ranges.append((i*batch_size, (i+1)*batch_size))

data_decrp = ["AI: Glob   User: Loc", "AI: Loc   User: Globe", "Both: Glob", "None: Globe",
                "AI: Glob   User: None", "AI: Loc   User: None", "AI: None   User: None",
                "AI: Glob   User: Combined", "AI: G|L    User: Glob", "AI: G|L    User: None",
                "AI: Combined   User: None", "All combined"]




for dt_dsc in data_decrp:
    if not os.path.exists(expr_path+"results_2/"+dt_dsc):
        os.makedirs(expr_path+"results_2/"+dt_dsc)
        #break


init_gp_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
user_data_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
agent_data_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
function_list = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]


trials_random = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
trials_greedy = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
trials_planning = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
trials_planning_knowledge = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]
trials_planning_all = [[[] for _ in range(len(trials_ranges))] for _ in range(len(users_set))]

## Methods, Trials, users, conditions
## Methods: PlanningAI, GreedyAI, RandomAI
table_entropy = np.zeros((3, 30, 4, 7))

def calc_entropy(trl):
    return 0


trl_count = 0


for i_exp in users_set:
    for i_rng, trl_rng in enumerate(trials_ranges):
        for i_trl in range(*trl_rng):

            if not os.path.exists(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_PlanningAI.pkl"):
                continue

            trl_count += 1
            
            trl_rand = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_RandomAI.pkl")
            trl_greed = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_GreedyAI_2.pkl")
            
            trl_plan = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_PlanningAI.pkl")
            """
            trl_plan_know = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_OmniPlanningAI.pkl")
            trl_plan_all = Trial(expr_path+"EXP_"+str(i_exp)+"/tiral_"+str(EXPERIMENT_ID_2)+"_"+str(i_trl)+"_OmniPlanningAI_2.pkl")
            #"""

            """
            if i_trl < 30:
                table_entropy[0, i_trl, i_exp, i_rng] = calc_entropy(trl_plan)
                table_entropy[1, i_trl, i_exp, i_rng] = calc_entropy(trl_greed)
                table_entropy[2, i_trl, i_exp, i_rng] = calc_entropy(trl_rand)
            """

            
            trials_random[i_exp][i_rng].append(trl_rand)
            trials_greedy[i_exp][i_rng].append(trl_greed)
            
            trials_planning[i_exp][i_rng].append(trl_plan)
            """
            trials_planning_knowledge[i_exp][i_rng].append(trl_plan_know)
            trials_planning_all[i_exp][i_rng].append(trl_plan_all)
            #"""

            init_gp_list[i_exp][i_rng].append(trl_greed.init_gp)
            user_data_list[i_exp][i_rng].append(trl_greed.user_data)
            agent_data_list[i_exp][i_rng].append(trl_greed.agent_data)
            function_list[i_exp][i_rng].append(trl_greed.function_df)
            
### ==================================================================================================
#print(a)

print("Total trials:", trl_count)

with open(expr_path+"run.log", "a") as f:
        f.write("Total trials:" + str(trl_count) + "\n\n")



#def plot_performance(my_users, my_rng, i_rng, score_idx=0, indices=np.arange(9), do_table=True, do_plot=True, draw_trajectories=False):
def plot_performance(my_users, my_rng, i_rng, score_idx=0, indices=np.arange(9), do_plot=True, draw_trajectories=False):
    params = [param_set[i] for i in my_users]
    caption = r'params ($\alpha$, $\beta$): ' + str(params) + '\n'
    caption += 'data: ' + data_decrp[i_rng]
    score_labels = ["optimization score", "Simple regret", "Normalized distance"]
    
    trls_random = [trl for i in my_users for j in my_rng for trl in trials_random[i][j]]
    trls_greedy = [trl for i in my_users for j in my_rng for trl in trials_greedy[i][j]]
    
    trls_planning = [trl for i in my_users for j in my_rng for trl in trials_planning[i][j]]
    trls_planning_know = [trl for i in my_users for j in my_rng for trl in trials_planning_knowledge[i][j]]
    trls_planning_all = [trl for i in my_users for j in my_rng for trl in trials_planning_all[i][j]]
    
    gp_list = [gp for i in my_users for j in my_rng for gp in init_gp_list[i][j]]
    usr_data_list = [dt for i in my_users for j in my_rng for dt in user_data_list[i][j]]
    agt_data_list = [dt for i in my_users for j in my_rng for dt in agent_data_list[i][j]]
    func_list = [func for i in my_users for j in my_rng for func in function_list[i][j]]

    n_trls = len(trls_greedy)
    print("Numbe of trials: ", n_trls, my_users, my_rng)
    if n_trls == 0:
        return

    scores_random = np.array([get_scores(trls_random[i])[score_idx] for i in range(n_trls)])
    scores_greedy = np.array([get_scores(trls_greedy[i])[score_idx] for i in range(n_trls)])
    
    scores_planning = np.array([get_scores(trls_planning[i])[score_idx] for i in range(n_trls)])
    
    scores_planning_know, scores_planning_all = None, None
    """
    scores_planning_know = np.array([get_scores(trls_planning_know[i])[score_idx] for i in range(n_trls)])
    scores_planning_all = np.array([get_scores(trls_planning_all[i])[score_idx] for i in range(n_trls)])
    #"""
    
    scores_single_random = np.array(random_baseline(func_list, n_arms, n_iters))
    scores_single_omnisci = np.array(omnisci_baseline(func_list, usr_data_list, agt_data_list, gp_list, n_arms, n_iters))
    scores_single_omnisci_user = np.array(omnisci_baseline(func_list, usr_data_list, None, gp_list, n_arms, n_iters))
    scores_single_omnisci_agent = np.array(omnisci_baseline(func_list, None, agt_data_list, gp_list, n_arms, n_iters))
    

    all_scores = [scores_single_random, scores_single_omnisci, scores_single_omnisci_agent,
                    scores_single_omnisci_user, scores_random, scores_greedy
                    , scores_planning_all
                    , scores_planning_know
                    , scores_planning]

    
    for i_scr in range(len(all_scores)):
        tmp_score = all_scores[i_scr]
        if tmp_score is not None:
            table_results[0, i_scr, my_users[0], i_rng] = tmp_score.shape[0]
            table_results[1, i_scr, my_users[0], i_rng] = np.mean(tmp_score[:,-1])
            table_results[2, i_scr, my_users[0], i_rng] = np.std(tmp_score[:,-1])
            #print("####  M", i_scr, table_results[:, i_scr, my_users[0], i_rng])


    legends = ["SA Random", "SA UCB", "UCB agent", "UCB user", "RandomAI", "GreedyAI",
                "PlanningAI + UK + UP",
                "PlanningAI + UK",
                "PlanningAI (proposed)"]    
    
    #"""
    if do_plot:
        plot_compare([all_scores[i] for i in indices], [legends[i] for i in indices],
                    expr_path+"results_2/"+data_decrp[i_rng]+"/"+score_labels[score_idx].split()[1]+"_users("+"-".join(str(x) for x in my_users)+")", 
                    indices, caption, score_labels[score_idx])
    """
    plot_compare([all_scores[i] for i in indices], [legends[i] for i in indices],
                    expr_path+"results_2/"+score_labels[score_idx].split()[1]+"_users("+"-".join(str(x) for x in my_users)+")", 
                    indices, caption, score_labels[score_idx])
    """
    
    #"""
    if draw_trajectories:
        plot_all_trials(trls_planning[:5], trls_greedy[:5], expr_path+"results/trajectories("+"_".join(str(x) for x in my_users)+
        #plot_all_trials(trls_random, trls_greedy, expr_path+"results/trajectories("+"_".join(str(x) for x in my_users)+
                            ")_data("+"-".join(str(x) for x in my_rng)+")")
    #"""




def plot_params(my_users, my_rng):
    params = [param_set[i] for i in my_users]
    datas = data_decrp[my_rng[-1]]
    caption = r'params ($\alpha$, $\beta$): ' + str(params) + '\n'
    caption += 'data: ' + str(datas)

    trls_planning = [[trl for j in my_rng for trl in trials_planning[i][j]] for i in my_users]
    trls_greedy = [[trl for j in my_rng for trl in trials_greedy[i][j]] for i in my_users]

    s_idx = 0

    scores_planning = np.array([[get_scores(trl)[s_idx] for trl in trials_set] for trials_set in trls_planning])
    scores_greedy = np.array([[get_scores(trl)[s_idx] for trl in trials_set] for trials_set in trls_greedy])
    
    #scores_planning = scores_planning[:,:,4:15:5].T
    #scores_greedy = scores_greedy[:,:,4:15:5].T
    
    #"""
    plot_compare(np.concatenate((scores_greedy, scores_planning)),
                    ["Greedy AI"+str(pars) for pars in params]+["Planning AI"+str(pars) for pars in params],
                    #["Greedy AI", "Greedy AI 2", "Planning AI"],
                    expr_path+"results/plot_params("+"-".join(str(x) for x in my_users)+
                    ")_data("+"-".join(str(x) for x in my_rng)+")_1", caption)
    #"""






"""
users_comb = [[0], [1], [2], [3], [4], [5],
                [0, 2, 4], [1, 3, 5],
                [0, 1], [2, 3], [4, 5],
                [0, 1, 2, 3, 4, 5]]
"""
users_comb = [[0], [1], [2], [3],
                [0, 1], [0, 2],
                [2, 3], [1, 3],
                [0, 1, 2, 3]]

rng_comb = [[0], [1], [2], [3], [4], [5], [6],
            [0, 2, 4], [1, 2], [4, 5], [4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6]]


## 1st index: [0]=No_of_trials, [1]=Mean, [2]=Std
## 2nd index: each methods as in the legends list
## 3rd index: users
## 4th index: conditions for prior data distribution
table_results = np.zeros((3, 9, 4, 7))

for usrs in users_comb[:4]:
    for i_rng, rng in enumerate(rng_comb[:7]):
        plot_performance(usrs, rng, i_rng, score_idx=0, do_plot=False)

np.save(expr_path+'results/table_results.npy', table_results)



#plot_performance([0], [0], i_rng=0, indices=np.arange(7))
#plot_performance([0, 2], [6], i_rng=6, indices=np.arange(7), draw_trajectories=True)

#plot_performance([0, 2], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))
#plot_performance([0, 2, 4], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))
#plot_performance([1, 3, 5], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))
#plot_performance([0,1,2,3,4,5], [0, 1, 2, 3, 4, 5], i_rng=-1, indices=np.arange(7))

#plot_performance([0], [0], 0, indices=np.arange(4), draw_trajectories=True)

#plot_performance([0,1,2,3,4,5], [0, 1, 2, 3, 4, 5, 6], i_rng=-1)

"""
for usrs in users_comb:
    i_rng = 6
    rng = rng_comb[i_rng]
    plot_performance(usrs, rng, i_rng)
#"""






#"""
for usrs in users_comb:
    for i_rng, rng in enumerate(rng_comb):
        #plot_performance(usrs, rng, i_rng)
        #plot_performance(usrs, rng, i_rng, indices=np.arange(7))
        #plot_performance(usrs, rng, i_rng, indices=[0,1,4,5,8], score_idx=1)
        
        #plot_performance(usrs, rng, i_rng, indices=[0,1,4,5,8], score_idx=0)
        break
        plot_performance(usrs, rng, i_rng, indices=[4,5,8], score_idx=0)
    break

#"""

#plot_performance([0], [0,1])


#for rng in rng_comb:
    #plot_params([4,5], rng)
#plot_params([1, 3, 5], [0, 1, 2])



#print("comparing the trials ...")
#compare_trials(scores_planning, scores_greedy, "res_"+expr)

#print("plotting all trials ...")
#plot_all_trials(trials_planning, trials_greedy, expr_path+"results/plot_compare_usr"+str(my_users)+"_trls"+str(my_rng))




"""
random_avg = np.mean(scores_random, axis=0)
omnisci_avg = np.mean(scores_omnisci, axis=0)
omniplanning_avg = np.mean(scores_planning, axis=0)
planning_avg = np.mean(scores_planning, axis=0)
greedy_avg = np.mean(scores_greedy, axis=0)

print()
print("UCB :", np.round(omnisci_avg,1))
print("Planning+UserKnowledge:", np.round(omniplanning_avg,1))
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
"""