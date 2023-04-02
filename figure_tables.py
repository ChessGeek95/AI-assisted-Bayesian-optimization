import numpy as np
from utils import PATH, get_scores


EXPERIMENT_ID = 143

expr_path = PATH+"trials/expriment_"+str(EXPERIMENT_ID)+"/"

table_tmp_fill = np.load(PATH+'trials/expriment_141/results/table_results.npy')

## Users: [(0.1, 0.2), (0.1, 0.7), (0.6, 0.2), (0.6, 0.7)]

## Methodes:    ["SA Random", "SA UCB", "UCB agent", "UCB user", "RandomAI", "GreedyAI",
##                "PlanningAI + UK + UP", "PlanningAI + UK", "PlanningAI (proposed)"] 

## 1st index: [0]=No_of_trials, [1]=Mean, [2]=Std
## 2nd index: each methods as in the legends list
## 3rd index: users
## 4th index: conditions for prior data distribution
table_results = np.load(expr_path+'results/table_results.npy')
table_results[:,:,0,:] = table_tmp_fill[:,:,0,:]
user_idx = [0,2,1,3]
table_results[:,:,:,:] = table_results[:,:,user_idx,:]

#data_decrp = (AI, User): { GL, LG, GG, LL, GN, LN, NN }




#cond_indx = [1, 2]
#tbl_2_mean = np.round(np.mean(table_results[1, method_idx, :, cond_indx], axis=2), 1)
#tbl_2_std = np.round(np.std(table_results[1, method_idx, :, cond_indx], axis=2), 1)

def print_table(tbl, tbl_std):
    for i in range(tbl.shape[0]):
        print(i+1, end='\t#\t')
        for j in range(tbl.shape[1]):
            print("$",tbl[i, j],'\pm',tbl_std[i, j],'$', end='\t')
            #print(tbl[i, j],'+-',tbl_std[i, j], end='\t')
        print()






################# Table 1 ####################
method_idx = [1,8,5,4,0]
tbl_1_mean = np.round(np.mean(table_results[1, method_idx, :, :], axis=2), 1)
#tbl_1_std = np.round(np.std(table_results[1, method_idx, :, :], axis=2), 1)
tbl_1_std = np.round(np.mean(table_results[2, method_idx, :, :], axis=2), 1)
print_table(tbl_1_mean, tbl_1_std)



################# Table 2 ####################
method_idx = [8,5]
tbl_2_mean = np.round(np.mean(table_results[1, method_idx, :, :], axis=2), 1)
tbl_2_std = np.round(np.std(table_results[1, method_idx, :, :], axis=2), 1)
#print_table(tbl_1_mean, tbl_1_std)

