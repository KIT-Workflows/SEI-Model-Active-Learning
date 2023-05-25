#!/bin/bash
#SBATCH --time=1-02:00:00
#SBATCH --partition=batch
#SBATCH --job-name=testX
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --array=0-10000%20
#SBATCH --output=Array_%x.%a.log
#SBATCH --cpus-per-task=5


#python abc.py $SLURM_ARRAY_TASK_ID

#Set the number of runs that each SLURM task should do
pwd; hostname; date

PER_TASK=5

# Calculate the starting and ending values for this task based
# on the SLURM task and the number of runs per task.
END_NUM=$(( ($SLURM_ARRAY_TASK_ID + 1) * $PER_TASK - 1 ))
START_NUM=$(( $SLURM_ARRAY_TASK_ID * $PER_TASK ))
sid=$SLURM_ARRAY_TASK_ID

# Print the task and run range
echo This is task $SLURM_ARRAY_TASK_ID, which will do runs $START_NUM to $END_NUM

for (( run=$START_NUM; run<=END_NUM; run++ )); do
  
  echo This is SLURM task $SLURM_ARRAY_TASK_ID, run number $run

  python 50kstartcopy.py $run

done

date
