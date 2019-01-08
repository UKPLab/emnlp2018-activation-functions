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
      # python3 /home/se55gyhe/Act_func/progs/doc_classification/01_NG/NG-basic.py penalized_tanh 1 Adadelta 3 0.6862848535205025 92 1.055588769388866 glorot_normal 90 4 newsgroup 0.5
#      functions = sys.argv[1]
#index=sys.argv[2]
#opt = sys.argv[3] # random.choice(['Adam',"RMSprop","Adagrad","Adadelta","Adamax","Nadam","sgd"])
#layers = int(sys.argv[4]) #random.choice([1, 2, 3, 4])
#dropout_values = float(sys.argv[5]) #random.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.75])
#filters = int(sys.argv[6]) # random.choice([50, 100, 200, 300, 400, 500])
#learning_rate = float(sys.argv[7]) # random.choice([0.002,0.001,0.005,0.01,0.1,0.5,1.0])
#init_name = sys.argv[8] # he_normal, rnormal, etc.
#embedding_dims = int(sys.argv[9])
#kernel_sizes = int(sys.argv[10])
#mydata = sys.argv[11] # TREC,SUBJ,MR,PE-sent2vec,PE-infersent
#dataset = dataset+mydata
#trainRatio = float(sys.argv[12])

      fun,_,opt,l,d,filters,learning_rate,recurrent_init,emb_dim,ks,_,datasize = w
      return w


hs={}
dir=sys.argv[-1]
allFiles=set()

for fn in os.listdir(dir):
  #if "glove" in fn: continue
  z = fn.split("/")[-1].split("__")
  # penalized_tanh__Adadelta__2__0.18463324736318487__3__he_uniform__0.7318047612943852__79__51__0.05-basic-acc1.csv
  # [fun,optimizer[0],str(layers),str(dropout_value),str(kernel_size),str(initializer[0]),str(learning_rate),str(filters),str(embedding_dims),datasize]
  opt,l,d,ks,recurrent_init,learning_rate,filters,emb_dim,datasize = z[1:10]
  datasize = datasize.split("-")[0]
  fle = (opt,l,d,ks,recurrent_init,learning_rate,filters,emb_dim,datasize)
  allFiles.add(fle)

runBatch=False
#runBatch=True

print(len(allFiles),len(sys.argv[1:-1]))

for fn in sys.argv[1:-1]:
  fun,_,opt,l,d,filters,learning_rate,recurrent_init,emb_dim,ks,_,datasize = readFile(fn) 
  #print(datasize)
  if (opt,l,d,ks,recurrent_init,learning_rate,filters,emb_dim,datasize) not in allFiles:
    print("{0} is missing".format(fn))
    if runBatch:
      os.system("sbatch %s"%(fn))

