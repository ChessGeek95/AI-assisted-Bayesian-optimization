import numpy as np
import matplotlib
import os
import argparse

from experiment_settings import ExperimentSettings
matplotlib.use('Agg')

from utils import PATH



if __name__ == "__main__":
    
    parser=argparse.ArgumentParser()

    parser.add_argument("-s", "--seed", help="Which seed?")
    parser.add_argument("-nt", "--n_trials", help="How many trials?")
    parser.add_argument("-na", "--n_arms", nargs="+",  help="Space dimension?")
    parser.add_argument("-nu", "--n_users",  help="How many various users?")
    parser.add_argument("-eid", "--expr_id", help="which experiment?")
    
    args=parser.parse_args()
     
    RANDOM_SEED  = eval(args.seed)
    N_TRIALS = eval(args.n_trials)
    N_ARMS = [eval(r) if isinstance(r, str) else r for r in args.n_arms]
    N_USERS = eval(args.n_users)
    EXPERIMENT_ID = eval(args.expr_id)

    
    EXP_PATH = PATH+"trials/expriment_"+str(EXPERIMENT_ID)+"/"
    experiment_sets = ["EXP_"+str(v) for v in range(N_USERS)]
    
    if not os.path.exists(EXP_PATH):
        os.makedirs(EXP_PATH)
        os.makedirs(EXP_PATH + "problems/")
        os.makedirs(EXP_PATH + "results/")
        for ex in experiment_sets:
            os.makedirs(EXP_PATH + ex + "/")

    #====================================================================
    print("EXPRIMENT ID:", EXPERIMENT_ID)
    print("generating new set of " + str(N_TRIALS) + " problems of " + str(N_ARMS) + "...")

    np.random.seed(RANDOM_SEED)
    
    expr_settings = ExperimentSettings()
    expr_settings.set_values(N_ARMS, N_TRIALS, id=EXPERIMENT_ID)
    expr_settings.save(path=EXP_PATH+"problems/problems.pkl")
