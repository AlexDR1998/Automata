#! /bin/sh
#$ -N ca_run
#$ -cwd
#$ -l h_rt=20:00:00
#$ -l h_vmem=4G

module load python/3.4.3
python ed_ca_run.py 8 $1