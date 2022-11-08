from sqlalchemy import func
from trial import Trial


import pathlib
import numpy as np
from utils import PATH



#PATH = str(pathlib.Path(__file__).parent.resolve())+'/'


expr = "exp_0"
trial_num = 1

#trial = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_PlanningAI.pkl")
#trial = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_GreedyAI.pkl")


trial_1 = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_PlanningAI.pkl")
trial_2 = Trial(PATH + "trials/"+expr+"/tiral_100_"+str(trial_num)+"_GreedyAI.pkl")

#print(type(trial))
#print(trial.user_data)
#print(trial.agent_data)
#print(trial.function_df[:5,:5])
print(trial_1.scores)
print(trial_2.scores)
#print(trial.agent_beliefs[4][0].shape)
#print(trial.user_beliefs[4][0].shape)
#print(trial.agent_beliefs[0][1].shape)
#print(trial.user_beliefs[0][1].shape)

trial_1.plot_compare_to(trial_2, expr+"_trl_"+str(trial_num))





