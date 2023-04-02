#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=300MB
#SBATCH --array=0-41
#SBATCH --output=slurm_files/run.log


SEED=987654321

N_JOBS=8

X_ARMS=50
Y_ARMS=50
N_TRIALS=350
N_ITERS=15
EXPERIMENT_ID=21






batch_size=$(( N_TRIALS/7 ))
user_idx=$(( SLURM_ARRAY_TASK_ID/7 ))
i_rng=$(( SLURM_ARRAY_TASK_ID%7 ))
trl_lr=$(( batch_size*(i_rng) ))
trl_ur=$(( batch_size*(i_rng+1)-1 ))



srun python3 main.py -s=$SEED -j=$N_JOBS -nt=$N_TRIALS -ni=$N_ITERS -eid=$EXPERIMENT_ID -na $X_ARMS $Y_ARMS -u=$user_idx -rng $trl_lr $trl_ur
