#!/bin/sh

#SBATCH --account=rrg-whitem
#SBATCH --mail-type=ALL
#SBATCH --mail-user=han8@ualberta.ca
#SBATCH --error=slurm-%j-%n-%a.err
#SBATCH --output=slurm-%j-%n-%a.out
#SBATCH --cpus-per-task=1
#SBATCH --mem=16000M
#SBATCH --time=0-02:00

cd ..
module load python/3.6
source $HOME/torch1env/bin/activate

python3 experiment_gui_r.py REM_Dyna 0.4 8 -1.0 1 10 1e-07 0.0 1 0.0 Q 0.0 0.0 0.0 4 1 10