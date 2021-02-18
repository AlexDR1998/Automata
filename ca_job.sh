#! /bin/sh
#$ -N ca_run
#$ -cwd
#$ -l h_rt=100:00:00
#$ -l h_vmem=4G

module load python/3.4.3
./ed_ca_run.py 8 $1