#! /bin/sh
#$ -N ca_run_rw
#$ -cwd
#$ -l h_rt=72:00:00
#$ -l h_vmem=6G
bash ca_job_rw.sh $SGE_TASK_ID