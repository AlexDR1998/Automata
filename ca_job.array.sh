#! /bin/sh
#$ -N ca_run
#$ -cwd
#$ -l h_rt=100:00:00
#$ -l h_vmem=4G
ca_job.sh $SGE_TASK_ID