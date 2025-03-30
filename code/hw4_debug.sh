#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=10G
#SBATCH --output=results/debug_%j_stdout.txt
#SBATCH --error=results/debug_%j_stderr.txt
#SBATCH --time=00:05:00
#SBATCH --job-name=hw4_debug
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw4/code/
#SBATCH --array=0

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

## SHALLOW
python hw4.py @exp.txt @oscer.txt @net_shallow.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --epochs 50 -vvv
