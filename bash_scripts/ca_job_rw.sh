#! /bin/sh
#$ -N ca_run_rw
#$ -cwd
#$ -l h_rt=72:00:00
#$ -l h_vmem=6G
# Initialise the environment modules
. /etc/profile.d/modules.sh

module load anaconda
source activate mphys_python
python ./ed_ca_random_walk.py 2 $1
python ./ed_ca_random_walk.py 3 $1
python ./ed_ca_random_walk.py 4 $1
python ./ed_ca_random_walk.py 5 $1
python ./ed_ca_random_walk.py 6 $1
python ./ed_ca_random_walk.py 7 $1
python ./ed_ca_random_walk.py 8 $1
source deactivate