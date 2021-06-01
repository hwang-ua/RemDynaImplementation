#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=han8@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000
#SBATCH --time=0-2:00

cd ..

#module load miniconda3
#source activate venv

module load python/3.6
source $HOME/torch1env/bin/activate
parallel :::: 'script/tasks_'"$SLURM_ARRAY_TASK_ID"'.sh'