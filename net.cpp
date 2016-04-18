#include <ctime>
#include <cstdlib>
#include <cmath>
#include "net.h"

Net::Net(int* s, int n_l) : shape(s), n_layers(n_l)
//constructor
//initilizes weights of network to random values between -1.0 and 1.0
{
  w_size = 0;
  srand((unsigned)time(0));
  rand();

  //determine number of weights needed
  for (int i=0; i < n_layers-1; i++)
    w_size += shape[i+1] + shape[i]*shape[i+1];

  //initiize weights to random values between -1.0 and 1.0
  weights = new double [w_size];
  for (int i=0; i < w_size; i++)
    weights[i] = 2.0*rand()/RAND_MAX - 1.0;
}

double* Net::output(const double* x) const
//takes in an array of values x of size shape[0] and returns the
//appropriate output array
//
//k = current working layer
//i = neuron i in layer k
//j = weight j for neuron i, j=0 gives the bias value
{
  int shift = 0;

  //copy initial input
  double* y = new double [shape[0]];
  for (int i=0; i < shape[0]; i++)
    y[i] = x[i];

  //calulate new layer outputs
  for (int k=0; k < n_layers-1; k++)
  {
    double* w_k = &(weights[shift]);
    double* temp = new double[shape[k+1]];

    //individual neuron values
    for (int i=0; i < shape[k+1]; i++)
    {
      temp[i] = w_k[i];
      for (int j=0; j < shape[k]; j++)
        temp[i] += w_k[shape[k+1]*(j+1) + i] * y[j];

      //take tanh for all neuron except final output neurons
      if (k < n_layers - 2)
        temp[i] = tanh(temp[i]);
    }

    //shift over to next layer's weights
    shift += shape[k+1] + shape[k]*shape[k+1];
    //replace inputs to layer with outputs from layer
    delete [] y;
    y = temp;
  }

  return y;
}

double* Net::w()
//array of weights
{
  return weights;
}
int Net::wLen()
//length of weights array
{
  return w_size;
}
