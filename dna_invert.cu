#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

//blockId -> the block number
//blockDim -> number of threads per block
//threadIdx -> thread number within block

//gid (1d) -> blockIdx.x * blockDim.x + threadIdx.x
//gid (2d) -> blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x * threadIdx.x

//random change

//read string from file into a vector -> translate chars to ints
void read_str(vector<int> &str, string file_name){
    ifstream input_stream (file_name);
    char c;
    int translated; 
    if(input_stream.is_open()){
      while(input_stream.get(c)) {
        if (c == 'A') {
          translated = 0;
        } else if(c == 'T') {
           translated = 1;
        } else if(c == 'G') {
           translated = 2;
        } else {
          translated = 3;
        }
        str.push_back(translated);    
      }
    }
    input_stream.close();
}

//print a sequence of characters to a file
void print_sequence_file(const int *sequence, int length, string file_name) {
  ofstream out_file;
  out_file.open (file_name, fstream::app);

  char chars[4] = {'A', 'T', 'G', 'C'}; 
  for(int i = 0; i < length; i ++) {
    out_file << chars[sequence[i]];
  }
  out_file << endl;
  out_file.close();
}

//invert a sequence, allocate on thread to each element of sequence
__global__ void invert(int *da, int N) {
  int tid = threadIdx.x;
  printf("tid is: %i\n", tid);

  int new_val; 
  if (da[tid] == 0) {
    new_val = 1;
  } if (da[tid] == 1) {
    new_val = 0;
  } if (da[tid] == 2) {
    new_val = 3;
  } if (da[tid] == 3) {
    new_val = 2;
  }

  da[tid] = new_val;
}

int main() {
  //INPUTS
  // int N = 8;

  //read into vector so for dynamic length + size checking
  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  //get the size
  int N = temp_sequence.size();

  cout << endl;
  //create host sequence array, device array
  int *ha = new int[N];
  int *da;

  //allocate device array using cuda malloc
  cudaMalloc((void **)&da, N*sizeof(int));

  //copy into array for vector sequence
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  //copy from host to device
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  //call kernel function
  invert<<<1,N>>>(da, N);    

  cudaDeviceSynchronize();

  cudaMemcpy(ha, da, N*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_sequence_file(ha, N, "output.txt");

  cudaFree(da);
  free(ha);
  return 0;
}