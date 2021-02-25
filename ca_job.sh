#! /bin/sh
#$ -N ca_run
#$ -cwd
#$ -l h_rt=72:00:00
#$ -l h_vmem=6G

module load anaconda
source activate mphys_python
python ed_ca_run.py 8 $1
source deactivate