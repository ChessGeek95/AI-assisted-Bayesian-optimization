import pathlib
import numpy as np
import matplotlib
import os

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

from interface import Interface
from users.simulated_user import GreedyUser
from agents.agents import EpsGreedyAgent, StrategicAgent

from utils import PATH

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)





def simulate(interface, user, agent, experiment_settings, n_trials=5, n_iters=10, name=None):
    scores = np.zeros((n_trials, n_iters))
    for episode in range(n_trials):
        trial = Trial()
        trial.set_values(user, agent, interface, experiment_settings, n_iters, episode)
        func_arg = experiment_settings.function_args[episode]
        start_point = experiment_settings.starting_points[episode]
        #print("#"*50, func_arg, "#"*5, start_point)
        user_data, agent_data = interface.reset(expr_settings.function_list[episode],
                                                expr_settings.user_data_list[episode],
                                                expr_settings.agent_data_list[episode],
                                                start_point)
        user.reset(user_data)
        agent.reset(agent_data, user_data)
        trial.add_belief(user.current_prediction(cur=False), agent.current_prediction(cur=False))
        #print(user.__class__.__name__, "prior size:", user_data[0].shape)
        #print(agent.__class__.__name__, "data size", agent_data[0].shape)
        #print()

        #print("settings ### m:", m, " start:", start_point)
        """
        if episode == 1000:
            pdf = PdfPages(PATH+'results/multipage_pdf.pdf')
        """
            

        for t in range(n_iters):

            #print(interface.cur, end=' ==> ')
            
            x_t = agent.take_action()
            #observation = interface.step(x_t, 0)
            #user.update(observation, x_t)
            #agent.update(observation)

            """ if episode == 0:
                
                gs = fig.add_gridspec(13,3)
                plot_interaction(user, agent, interface, 0, 0, fig, gs, t)
             """
            #print(interface.cur, end=' -> ')

            y_t = user.take_action(agent_action=x_t)
            observation = interface.step(x_t, y_t)
            agent.update(observation, y_t)
            user.update(observation)

            score = interface.get_score()
            scores[episode, t] = score[1]

            """
            if episode == 1000:
                fig = plt.figure(figsize=(25, 21))
                gs = fig.add_gridspec(13,3)
                plot_interaction(user, agent, interface, 1, 1, fig, gs, t)
                #fig.tight_layout()
                plt.subplots_adjust(left=0.1,
                                bottom=0.1, 
                                right=0.9, 
                                top=0.9, 
                                wspace=0.4, 
                                hspace=0.4)
                pdf.savefig(fig)
                plt.close()
            """
            user_belief = user.current_prediction(cur=False)
            agent_belief = agent.current_prediction(cur=False)
            trial.add_belief(user_belief, agent_belief)

            #print(t,' ', interface.cur, ' ->\t',np.round(interface.z_queries[-1],3),'\t', score[1])

        """
        if episode == 1000:
            #fig.colorbar()
            pass
            pdf.close()
        """

        print('$$$ ', episode+1, '=>', score[1])
        trial.add_queries(interface.xy_queries, interface.z_queries, scores[episode])
        trial.save(path=EXP_PATH+"tiral_"+str(experiment_settings.id)+"_"
                                            +str(episode)+"_"
                                            +name
                                            +".pkl")
    
    print('-'*25)
    print("Mean:", np.round(np.mean(scores[:,-1]),2))
    #print("#"*50, '\n')
    # evaluate
    return scores







if __name__ == "__main__":

    #=== init params
    N_TRIALS = 2
    N_ITERS = 10
    THETA_U = (.2, 0.5)
    N_ARMS = (50, 50)
    GENERATE_NEW_EXP = True
    EXP_PATH = PATH+"trials/exp_25/"
    
    
    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH)
    

    expr_settings = ExperimentSettings()
    if GENERATE_NEW_EXP:
        expr_settings.set_values(N_ARMS, N_TRIALS, id=100)
        
        if not os.path.exists(EXP_PATH + 'exp_settings'):
            os.makedirs(EXP_PATH + 'exp_settings')
        expr_settings.save(path=EXP_PATH+"exp_settings/exp_setting.pkl")
    else:
        expr_settings.load(path=EXP_PATH+"exp_settings/exp_setting.pkl")
    #-----------------------
    
    #=== init the environment
    interface = Interface(N_ARMS)
    #interface.reset()
   #print(np.round(interface.function, 2))
    #print("-"*50)
    #print(interface.all_points[:5])
    
    #=== init the synthetic user
    kernel = ConstantKernel(5, constant_value_bounds="fixed") * RBF(10, length_scale_bounds="fixed")
    gp_model = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-3)
    user = GreedyUser(gp_model, N_ARMS, interface.get_cur(), THETA_U)

    #=== init the AI agent
    kernel, gp_model = deepcopy(kernel), deepcopy(gp_model)
    greedy_agent = EpsGreedyAgent(gp_model, N_ARMS, interface.get_cur())
    strategic_agent = StrategicAgent(gp_model, N_ARMS, interface.get_cur())
    
    #=== run the experiment
    np.random.seed(987654321)
    scores_baseline = simulate(interface, user, greedy_agent, expr_settings, N_TRIALS, N_ITERS, name="GreedyAI")
    scores_my_method = simulate(interface, user, strategic_agent, expr_settings, N_TRIALS, N_ITERS, name="PlanningAI")
    #"""
    #np.save(PATH + 'results/base.npy', scores_baseline)
    #np.save(PATH + 'results/mine.npy', scores_my_method)

    print('#'*80)
    #print(scores_baseline)
    print('='*80)
    #print(scores_my_method)
    print('#'*80)

    #plot_compare(scores_baseline, scores_my_method, ["baseline", "our method"])
    #"""S