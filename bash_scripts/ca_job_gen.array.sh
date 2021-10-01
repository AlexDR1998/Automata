#! /bin/sh
#$ -N ca_run_gen
#$ -cwd
#$ -l h_rt=72:00:00
#$ -l h_vmem=6G
bash ca_job_gen.sh $SGE_TASK_ID