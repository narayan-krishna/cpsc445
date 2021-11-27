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

string reverse_translate(int triplet_index) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 

  string combo;
  combo += chars[triplet_index/16];
  triplet_index = triplet_index % 16;

  combo += chars[triplet_index/4];
  triplet_index = triplet_index % 4;

  combo += chars[triplet_index];

  return combo;
}
  
void print_results_file(const int *combo_counter, string file_name) {
  cout "here" << endl;
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(int i = 0; i < 64; i ++) {
    if (combo_counter[i] > 0) {
      out_file << reverse_translate(i)  << " " << combo_counter[i] << endl;
    }
  }
  out_file.close();
}

//the sequence da, sequence length, n
__global__ void parse(int *da, int *dcounter, int N) {
  int tid = threadIdx.x;
  int offset_loc = tid*3;

  printf("tid is: %i\n", tid);

  int loc_store = 0;
  loc_store += da[offset_loc] * 1;
  loc_store += da[offset_loc + 1] * 4;
  loc_store += da[offset_loc + 2] * 16;

  //translate the number combination into number count
  printf("loc_store is: %i\n", loc_store);

  atomicAdd(&dcounter[loc_store], 1);
}

int main() {
  //INPUTS
  // int N = 8;

  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  int N = temp_sequence.size();
  cout << N << endl;
  int divisible = N % 3;
  cout << "num elements to be added: " << 3 - divisible << endl;
  if (divisible != 0) { temp_sequence.resize(N + (3 - divisible)); }
  N = temp_sequence.size();
  cout << N << endl;

  int *ha = new int[N];
  int *hcounter = new int[64]{0};
  int *da, *dcounter;

  cudaMalloc((void **)&da, N*sizeof(int));
  cudaMalloc((void **)&dcounter, 64*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da
  cudaMemcpy(dcounter, hcounter, 64*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  // int W = 16; //establish thread count
  // reduce_sum<<<1,W>>>(da, N); //call reduce sum using 1 block, 16 threads

  // parse<<<1,N/3>>>(da, dcounter, N);    
  parse<<<1,1>>>(da, dcounter, N);    

  cudaDeviceSynchronize();

  // int sum; //sum in parallel
  cudaMemcpy(hcounter, dcounter, 64*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_results_file(hcounter, "output.txt");

  // int expected_sum = (N-1)*N*(2*N-1)/6;
  // printf("%i (should be %i)", sum, expected_sum); //print sum
  free(da);
  cudaFree(da);

  free(ha);
  free(hcounter);

  return 0;
}