#!/bin/bash
#SBATCH -t 72:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1GB
#SBATCH --array=0-41
#SBATCH --output=slurm_files/test_run.log


#126-167

SEED=987654321


N_JOBS=5



X_ARMS=50
Y_ARMS=50
N_TRIALS=210
N_ITERS=20
EXPERIMENT_ID=143

N_PARTS=42

NNN=1
my_job_id=$(( SLURM_ARRAY_TASK_ID/NNN ))
#wating=$(( SLURM_ARRAY_TASK_ID - 119 ))

batch_size=$(( N_TRIALS/N_PARTS ))
user_idx=$(( SLURM_ARRAY_TASK_ID/N_PARTS ))
i_rng=$(( SLURM_ARRAY_TASK_ID%N_PARTS ))
trl_lr=$(( batch_size*(i_rng) ))
trl_ur=$(( batch_size*(i_rng+1)-1 ))

#sleep $wating
echo "job:$my_job_id   u:$user_idx  $batch_size   rng:$i_rng $trl_lr $trl_ur" >> slurm_files/test.log

srun python3 main.py -s=$SEED -j=$N_JOBS -nt=$N_TRIALS -ni=$N_ITERS -eid=$EXPERIMENT_ID -na $X_ARMS $Y_ARMS -u=$user_idx -rng $trl_lr $trl_ur
