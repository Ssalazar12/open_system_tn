#!ttusr/bin/bash

# name of queue job
#$ -N test

# write errors and logfiles in folder log/
# folder MUST EXIST!
#$ -e logs/ # give absolut path to the desired location
#$ -o logs/
#$ -l h_vmem=35G
#$ -pe smp 10
# start program, if excecutable of course just call it
export JULIA_NUM_THREADS=10
module load julia
julia --threads 10 /home/user/santiago.salazar-jaramillo/open_system_tn/scripts/kerr_cavities.jl
