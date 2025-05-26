#/bin/bash
## Code to run all the jobs for this codebase 

# sbatch --array=1-3 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1' launchGPU.sh
# sbatch --array=1-3 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1' launchGPU.sh
# sbatch --array=1-4 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1 --train_icm' launchGPU.sh
sbatch --array=1-3 --export=ALL,ALG='ppo.py',ARGSS='--env_name Craftax-Symbolic-v1 --ent_coef=0.1' launchGPU.sh
# sbatch --array=1-3 --export=ALL,ALG='ppo.py',ARGSS='--env_name Craftax-Symbolic-v1 --ent_coef=0.1 --use_e3b --train_icm --icm_reward_coeff=0' launchGPU.sh
# sbatch --array=1-3 --export=ALL,ALG='ppo_rnd.py',ARGSS='--env_name Craftax-Symbolic-v1 --ent_coef=0.1 --use_rnd' launchGPU.sh

