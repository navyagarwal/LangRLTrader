#!/bin/sh
### Set the job name (for your reference)
#PBS -N nllb54b-a2c-risk
### Set the project name, your department code by default
#PBS -P misn_mota.spons
### Request email when job begins and ends
#PBS -m bea
### Specify email address to use for notification.
#PBS -M $USER@iitd.ac.in
####
#PBS -l select=1:ncpus=01:mem=100gb:centos=icelake
### Specify "wallclock time" required for this job, hhh:mm:ss
#PBS -l walltime=60:00:00

##PBS -l software=
# After job starts, must goto working directory. 
# $PBS_O_WORKDIR is the directory from where the job is fired. 
#PBS -o logs/a2c_risk_output.log
#PBS -e logs/a2c_risk_error.log
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo "conda env activating"
source /home/apps/skeleton/condaBaseEnv
conda activate /home/ee/phd/eez228470/anaconda3/envs/greengraphs
python3 rl-agents/a2c_train_risk.py