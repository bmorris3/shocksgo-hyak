s = """#!/bin/bash
## Job Name
#SBATCH --job-name=test-job
## Allocation Definition
#SBATCH --account=stf
#SBATCH --partition=stf
## Resources
## Nodes
#SBATCH --nodes=1
## Tasks per node (Slurm assumes you want to run 28 tasks, remove 2x # and adjust parameter if needed)
#SBATCH --ntasks-per-node=1
## Walltime (two hours)
#SBATCH --time=1:00:00
# E-mail Notification, see man sbatch for options

##turn on e-mail notification
##SBATCH --mail-type=ALL
##SBATCH --mail-user=brettmorris21@gmail.com

## Memory per node
#SBATCH --mem=10G
## Specify the working directory for this job
#SBATCH --workdir=/usr/lusers/bmmorris/git/shocksgo-hyak/

python run.py {0}
"""
import os
import numpy as np

periods = [5, 88, 225, 250, 275, 300, 365.25, 400, 425, 450, 475, 500, 687, 11.86*365.25, 29.457*365.25, 200, 175, 150, 125, 100, 350, 75]

for period in periods:
    with open('batch_{0}.sh'.format(period), 'w') as w: 
        w.write(s.format(period))
    os.system('sbatch -p ckpt -A stf-ckpt {0}'.format('batch_{0}.sh'.format(period)))


