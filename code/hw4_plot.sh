#!/bin/bash
#
#SBATCH --partition=gpu
#SBATCH -x c860
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=40
#SBATCH --mem=20G
#SBATCH --output=results/plot_%j_stdout.txt
#SBATCH --error=results/plot_%j_stderr.txt
#SBATCH --time=00:20:00
#SBATCH --job-name=aml_plot
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw4/code/

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

python plot.py -v @exp.txt @oscer.txt
