acts="penalized_tanh swish sin sigmoid tanh relu elu selu linear maxsig cosper minsin maxtanh cube tanhrev"

for act in ${acts}; do
  sh gen.sh $act 1;
done

### just 1p of data

for act in ${acts}; do
  sh gen.sh $act 1 0.01;
done

