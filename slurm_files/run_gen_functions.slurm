#!/bin/bash
#SBATCH -t 00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=500MB
#SBATCH --output=slurm_files/run_gen_functions.log


SEED=987654321

N_JOBS=8

X_ARMS=50
Y_ARMS=50
N_TRIALS=210
N_ITERS=20
EXPERIMENT_ID=141

N_USERS=4


srun python3 main_generate_functions.py -s=$SEED -nt=$N_TRIALS -nu=$N_USERS -eid=$EXPERIMENT_ID -na $X_ARMS $Y_ARMS

