#!/bin/bash
## Run with sbatch --array=1-3 launch.sh

#SBATCH --partition=long                                # Ask for unkillable job
#SBATCH --cpus-per-task=8                               # Ask for 2 CPUs
#SBATCH --ntasks=1                                      # Ask for 2 tasks per GPU 
#SBATCH --gres=gpu:1                                    # Ask for GPUs
#SBATCH --mem=64G                                       # Ask for 10 GB of RAM
#SBATCH --time=11:55:00                                  # The job will run for 3 hours

#SBATCH -o /network/scratch/g/glen.berseth/slurm-%j.out  # Write the log on scratch

module load miniconda/3
conda activate jaxrl

echo $ALG
echo $ENV_ID

## srun is needed to run the multiple tasks per GPU
python $ALG --seed $SLURM_ARRAY_TASK_ID $ARGSS
