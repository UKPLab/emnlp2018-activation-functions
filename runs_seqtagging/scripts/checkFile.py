import sys,os

# [datasetName, word_embeddings, recurrent_init, opt, str(l), str(d), opt, str(size[0]), str(learning_rate), str(datasize), act_flag, 'f1.csv'] 

# python3 /home/se55gyhe/Act_func/sequence_tagging/arg_min/PE-my.py swish 357 Adamax 2 0.455495693525503 0.0015707497929066331 orth 0.3
# /work/scratch/se55gyhe/Act_func/seq_tag/ArgMinPE/results/relu/pe__levy_deps.words__varscaling__Adamax__4__0.5270001466090909__Adamax__181__0.0016232990964943923__0.3__None__f1.csv

# SAMPLE USAGE:
# python3 checkFile.py /work/scratch/se55gyhe/Act_func/seq_tag/ArgMinPE/results/swish/pe__levy_deps.words__*csv .|grep 0.05

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
  if not fn.endswith(".sh"): continue
  fun,size,opt,l,d,learning_rate,recurrent_init,datasize = readFile(dir+"/"+fn)
  hs[(size,opt,l,d,learning_rate,recurrent_init,datasize)] = fn
  allFiles.add(fn)

runBatch=False
runBatch=True

for ifn,fn in enumerate(sys.argv[1:-1]):
  done = False
  #print("-->",fn,ifn)
  try:
   test_score,dev_score = extract(fn)
   z = fn.split("/")[-1].split("__")
   #print(z); sys.exit(1)
   # ['pe', 'levy_deps.words', 'glorot_normal', 'Adagrad', '3', '0.141387332436817', 'Adagrad', '366', '0.009985772367561775', '0.05', 'None', 'f1.csv']
   size,opt,l,d,learning_rate,recurrent_init,datasize = z[7],z[3],z[4],z[5],z[8],z[2],z[9] 
   filn = hs[(size,opt,l,d,learning_rate,recurrent_init,datasize)]
   if filn in allFiles:
     allFiles.remove(filn)
     #print("REMOVED",filn)
   done = True
   #print(filn,done,ifn)
  except ValueError:
   #sys.stderr.write("\tError: %s\n"%fn)
   pass
  except TypeError:
   #sys.stderr.write("\tNot yet done: %s\n"%fn)
   pass  

  if done==False:
    z = fn.split("/")[-1].split("__")
    #print(z[2:])
    recurrent_init,opt,l,d,_,size,learning_rate,datasize = z[2:-2]
    fn = hs[(size,opt,l,d,learning_rate,recurrent_init,datasize)]
    print(fn,"hasn't finished")
    if runBatch:
      os.system("sbatch %s/%s"%(dir,fn))
    allFiles.remove(fn)

#for file in allFiles:
#  print(file,"hasn't been started")
