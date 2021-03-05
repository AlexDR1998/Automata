#! /bin/sh
#$ -N ca_run_6
#$ -cwd
#$ -l h_rt=72:00:00
#$ -l h_vmem=6G
# Initialise the environment modules
. /etc/profile.d/modules.sh

module load anaconda
source activate mphys_python
python ./ed_ca_run.py 6 $1
source deactivate