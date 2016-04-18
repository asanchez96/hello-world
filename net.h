#ifndef NET_H
#define NET_H

class Net
{
public:
  Net(int* shape, int num_layers=3);

  double* output(const double* x) const;
  double* w();
  int wLen();

private:
  double* weights;
  int w_size;
  int* shape;
  int n_layers;
};

#endif


/*
in markov chain program only use one term of the full potential to move forward
however when you test the two final points, test the full prob

(move through the targets sequencially) OR move through them in a random order

use one target for each set of leapfrog steps, change targets when you change
direction

!!!Investigate Multinest!!!

make a github account
*/
