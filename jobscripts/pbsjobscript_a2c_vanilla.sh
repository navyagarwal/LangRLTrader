#!/bin/sh
### Set the job name (for your reference)
#PBS -N nllb54b-lora
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
#PBS -o logs/a2c_vanilla_output.log
#PBS -e logs/a2c_vanilla_error.log
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
echo "conda env activating"
source /home/apps/skeleton/condaBaseEnv
conda activate /home/ee/phd/eez228470/anaconda3/envs/greengraphs
python3 rl_agents/a2c_vanilla.py

#job 
#time -p mpirun -n {n*m} executable
#NOTE
# The job line is an example : users need to change it to suit their applications
# The PBS select statement picks n nodes each having m free processors
# OpenMPI needs more options such as $PBS_NODEFILE
