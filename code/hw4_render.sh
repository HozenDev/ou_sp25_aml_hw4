#!/bin/bash
#
#SBATCH --partition=debug
#SBATCH --cpus-per-task=20
#SBATCH --mem=10G
#SBATCH --output=results/hw3_render_%j_stdout.txt
#SBATCH --error=results/hw3_render_%j_stderr.txt
#SBATCH --time=00:10:00
#SBATCH --job-name=hw3_render
#SBATCH --mail-user=Enzo.B.Durel-1@ou.edu
#SBATCH --mail-type=ALL
#SBATCH --chdir=/home/cs504305/hw3/code/

#################################################
# Do not change this line unless you have your own python/tensorflow/keras set up
. /home/fagg/tf_setup.sh
conda activate dnn

## SHALLOW
python hw3_base.py @net_deep_only_render.txt @exp.txt -vvv --cache "" --render --nogo
