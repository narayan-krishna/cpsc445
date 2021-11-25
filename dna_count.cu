#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

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

void print_results_file(const int *char_counter, string file_name) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(int i = 0; i < 4; i ++) {
    out_file << chars[i] << " " << char_counter[i] << endl;
  }
  out_file.close();
}

//the sequence da, sequence length, n
__global__ void count(int *da, int *dcounter, int N) {
  int tid = threadIdx.x;
  // printf("tid is: %i\n", tid);
  dcounter[da[tid]] ++;
}

int main() {
  //INPUTS
  // int N = 8;

  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  int N = temp_sequence.size();

  cout << endl;
  int *ha = new int[N];
  int *hcounter = new int[4]{0};

  int *da, *dcounter;
  cudaMalloc((void **)&da, N*sizeof(int));
  cudaMalloc((void **)&dcounter, 4*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da
  cudaMemcpy(dcounter, hcounter, 4*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  // int W = 16; //establish thread count
  // reduce_sum<<<1,W>>>(da, N); //call reduce sum using 1 block, 16 threads

  count<<<1,N>>>(da, dcounter, N);    

  cudaDeviceSynchronize();

  // int sum; //sum in parallel
  cudaMemcpy(hcounter, dcounter, N*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_results_file(hcounter, "output.txt");

  // int expected_sum = (N-1)*N*(2*N-1)/6;
  // printf("%i (should be %i)", sum, expected_sum); //print sum
  cudaFree(da);
  free(ha);
  return 0;
}