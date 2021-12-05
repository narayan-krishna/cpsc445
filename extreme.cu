#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

//read string from file into a vector -> translate chars to ints
void read_csv(vector<float> &values, const string &path/*, int &new_line_loc*/){
    ifstream input_stream (path);

    if (!input_stream.is_open()) {
      cerr << "coudn't find/open file..." << endl;
      exit(EXIT_FAILURE);
    }

    for(string line; getline(input_stream, line);) {
      stringstream ss(line);

      string float_string;
      while(getline(ss, float_string, ',')) {
        values.push_back( (float)atof(float_string.c_str()) );
      }
    }
}

//print a sequence of characters to a file
void print_to_csv(const float *sequence, int length, string output_file) {

  ofstream out_file;
  out_file.open (output_file);

  for(int i = 0; i < length; i ++) {
    out_file << sequence[i] << endl;
  }

  out_file.close();
  // system("head output.csv");
  // system("rm output.csv");
}

__global__ 
void sqrt(float *da, int N) {

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ float s[512];
  //allocated 512 floats per block
  //copy over 512 floats into the corresponding block

  s[tid] = da[gid];
  __syncthreads();
  da[gid] = sqrt(s[tid]);
  // printf("tid is: %i, seeing value: %f\n", tid, da[tid]);
}

int main() {
  cout << "\ncsv input head --------------------" << endl;
  system("head input.csv");
  cout << "\n-----------------------------" << endl;

  //INPUTS
  int N = 0;

  //read into vector so for dynamic length + size checking
  vector<float> inputs;
  read_csv(inputs, "input.csv");
  cout << "\ninputs(" << inputs.size() << ")" << endl;

  N = inputs.size();

  float *ha = new float[N];

  for(int i = 0; i < N; i++) {
    ha[i] = inputs[i];
  }

  for(int i = 0; i < 20; i ++) {
    cout << inputs[i] << ", ";
  }
  cout << "..." << endl; cout << endl;

  float *da;
  cudaMalloc((void **) &da, N*sizeof(float));
  cudaMemcpy(da, ha, N*sizeof(float), cudaMemcpyHostToDevice); //copy ints from ha into da

  int Nthreads = 512;
  int Nblocks = (N + (Nthreads - 1)) / Nthreads;
  cout << Nthreads << ", " << Nblocks << endl;
  sqrt<<<Nblocks,Nthreads>>>(da, N);
  cudaDeviceSynchronize();

  cudaMemcpy(ha, da, N*sizeof(float), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_to_csv(ha, N, "output.csv");

  cout << "head output csv" << "--------------" << endl;
  system("head output.csv");
  cout << "------------------------------------" << endl;

  cudaFree(da);
  free(ha);
  return 0;
}