#!/bin/sh

#SBATCH --account=def-amw8
#SBATCH --mail-type=ALL
#SBATCH --mail-user=han8@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=8000
#SBATCH --time=0-00:30

cd ..
#module load miniconda3
#source activate venv

module load python/3.6
source $HOME/torch1env/bin/activate
python3 -u random_walk_data_cgw.py
