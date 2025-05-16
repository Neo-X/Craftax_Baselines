#/bin/bash
## Code to run all the jobs for this codebase 

# sbatch --array=1-3 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1' launchGPU.sh
# sbatch --array=1-3 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1' launchGPU.sh
sbatch --array=1-4 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1 --train_icm' launchGPU.sh
sbatch --array=1-4 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1 --use_e3b' launchGPU.sh

