import sys,os

# SAMPLE USAGE:
# python3 ~/Act_func/sequence_tagging/arg_min/runs/scripts/checkMissing.py ~/Act_func/sequence_tagging/arg_min/runs/scripts/tanh/*.sh /work/scratch/se55gyhe/Act_func/seq_tag/POS/de/results/tanh
 
#  get the batches that have never been run


def extract(fn):
  doRead=False
  for line in open(fn):
    line = line.strip()
    if doRead:
      test,dev = line.split("\t")
      t = [float(x) for x in test.split(",")[1:6]]
      return t,[float(x) for x in dev.split(",")]
    if line.startswith("Units") or line.startswith("Filters"): 
      doRead=True

def readFile(fn):
  for line in open(fn):
    line = line.strip()
    if line.startswith("python3"):
      x = line.split()
      w = x[2:]
      fun,size,opt,l,d,learning_rate,recurrent_init,datasize = w
      return w


hs={}
dir=sys.argv[-1]
allFiles=set()

for fn in os.listdir(dir):
  #if "glove" in fn: continue
  z = fn.split("/")[-1].split("__")
  #pe__levy_deps.words__varscaling__Adamax__4__0.5270001466090909__Adamax__181__0.0016232990964943923__0.3__None__f1.csv
  recurrent_init,opt,l,d,_,size,learning_rate,datasize = z[2:10]
  fle = (recurrent_init,opt,l,d,size,learning_rate,datasize)
  allFiles.add(fle)

runBatch=False
#runBatch=True

for fn in sys.argv[1:-1]:
  fun,size,opt,l,d,learning_rate,recurrent_init,datasize = readFile(fn) 
  if (recurrent_init,opt,l,d,size,learning_rate,datasize) not in allFiles:
    print("{0} is missing".format(fn))
    if runBatch:
      os.system("sbatch %s"%(fn))

