#!/bin/bash
#
#SBATCH --partition=gpu_a100
#SBATCH --nodelist=c908
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=30G
#SBATCH --output=results/deep_%j_stdout.txt
#SBATCH --error=results/deep_%j_stderr.txt
#SBATCH --time=02:00:00
#SBATCH --job-name=hw4_deep
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw4/code/
#SBATCH --array=0-4

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

## SHALLOW
python hw4.py -vvv @exp.txt @oscer.txt @net_deep.txt --exp_index $SLURM_ARRAY_TASK_ID --cpus_per_task $SLURM_CPUS_PER_TASK --save_model --render --cache ""
