acts="penalized_tanh swish sin sigmoid tanh relu elu selu linear maxsig cosper minsin maxtanh cube tanhrev lrelu001 lrelu030"

for act in ${acts}; do
  sh gen.sh $act 1 0.3;
done

### just 1p of data

for act in ${acts}; do
  sh gen.sh $act 1 0.05;
done

