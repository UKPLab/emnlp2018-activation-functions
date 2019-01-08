#! /usr/bin/python


import sys

act=sys.argv[1]
index=sys.argv[2]
cmd=sys.argv[3]

if "glove" in cmd:
  mem = 6000
else:
  mem = 2000

text="""#!/bin/bash
 
#SBATCH -J Act_%s_%s
#SBATCH --mail-user=eger@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=FAIL
#SBATCH -e /work/scratch/se55gyhe/log/output.err.%%j
#SBATCH -o /work/scratch/se55gyhe/log/output.out.%%j

#SBATCH -n 1 # Number of cores
#SBATCH --mem-per-cpu=%d
#SBATCH -t 23:59:00 # Hours, minutes and seconds, or '#SBATCH -t 10' -only mins
#module load intel python/3.5

%s
"""%(act,index,mem,cmd)


print text
