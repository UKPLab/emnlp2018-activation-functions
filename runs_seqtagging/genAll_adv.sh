acts="penalized_tanh swish sin sigmoid tanh relu elu selu linear maxsig cosper minsin maxtanh cube tanhrev"
acts="leakyrelu-0.01 leakyrelu-0.3 prelu maxout-3 maxout-4 maxout-2"

for act in ${acts}; do
  sh gen.sh $act 1 0.5;
done

for act in ${acts}; do
  sh gen.sh $act 1 0.05;
done

