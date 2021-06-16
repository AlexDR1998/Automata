#! /bin/sh
#$ -N ca_run_5
#$ -cwd
#$ -l h_rt=72:00:00
#$ -l h_vmem=6G
bash ca_job_5.sh $SGE_TASK_ID