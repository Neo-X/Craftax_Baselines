#/bin/bash
## Code to run all the jobs for this codebase 

sbatch --array=1-3 --export=ALL,ARGSS='--env_name Craftax-Symbolic-v1' launchGPU.sh

