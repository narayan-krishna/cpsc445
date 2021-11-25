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

// __global__ void reduce_sum(int * da, int N) {
//     //array a [1,2,3,...,10,11,12]
//   int W = blockDim.x; 
//   int tid = threadIdx.x;
//   for(int i=tid+W; i<N; i+=W) da[tid]+=da[i];
//   __syncthreads();

// //__shared__ int tmp[1024];
// //   tmp[gid] = da[gid];


//   for(int delta=1; delta<W; delta*=2) { //set to 1, then set to 2 (w is 4 and
//     int i = tid*2*delta;                // delta doubles over loop
//     if (i + delta < N) {
//       da[i] += da[i+delta];
//       printf("%i (%i): %i\n", i, delta, da[i]);
//     }
//     __syncthreads();
//   }
// }
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

//the sequence da, sequence length, n
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

  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  int N = temp_sequence.size();

  cout << endl;
  int *ha = new int[N];
  int *da;
  cudaMalloc((void **)&da, N*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  // int W = 16; //establish thread count
  // reduce_sum<<<1,W>>>(da, N); //call reduce sum using 1 block, 16 threads

  invert<<<1,N>>>(da, N);    

  cudaDeviceSynchronize();

  // int sum; //sum in parallel
  cudaMemcpy(ha, da, N*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_sequence_file(ha, N, "output.txt");

  // int expected_sum = (N-1)*N*(2*N-1)/6;
  // printf("%i (should be %i)", sum, expected_sum); //print sum
  cudaFree(da);
  free(ha);
  return 0;
}