#train=/work/scratch/se55gyhe/CCP_FINAL/${ds}/data/FastText/1_1/${part}.train.label
act_func=$1
index=$2

ds=PE-infersent
ratio=$3

for i in `seq -s ' ' 1 200`; do
  echo $i
  cmd=`sed -n ${i},${i}p ../hyperparams_sentenceclassification`
  echo $cmd
  fullcmd="python3 /home/se55gyhe/Act_func/progs/meta.py $act_func $index $cmd ${ds} ${ratio}"
  mkdir -p scripts/${act_func}
  python generateBatch.py  ${act_func} ${index} "${fullcmd}" > scripts/${act_func}/${act_func}_${index}_${i}_${ratio}.sh
done
