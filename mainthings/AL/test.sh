#!/bin/bash
#SBATCH --job-name=Lizj
#SBATCH --output=jobout_Liz.out
#SBATCH --error=joberr_Liz.err
#SBATCH --partition=batch
#SBATCH --time=0-10:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1

pwd; hostname; date

for i in {500..4000..500}
do
  echo "Welcome $i times"
done

date