#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;

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

//print counts to file
void print_results_file(const int *char_counter, string file_name) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(int i = 0; i < 4; i ++) {
    out_file << chars[i] << " " << char_counter[i] << endl;
  }
  out_file.close();
}

//each thread checks corresponding element, and
//adds to the counter (atomicAdd) depending on what
//it is
__global__ void count(int *da, int *dcounter, int N) {
  int tid = blockDim.x * blockIdx.x + threadIdx.x;
  // printf("tid is: %i\n", tid);
  printf("%i", da[tid]);
  // dcounter[da[tid]] = dcounter[da[tid]] + 1;
  atomicAdd(&dcounter[da[tid]], 1);
}

int main() {

  //read into vector so for dynamic length + size checking
  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  //get the size
  int N = temp_sequence.size();

  cout << endl;
  //create host sequence array, device array
  int *ha = new int[N];
  //this is only 4 ints, one for each type of char
  int *hcounter = new int[4]{0};

  //allocate in device memory
  int *da, *dcounter;
  cudaMalloc((void **)&da, N*sizeof(int));
  cudaMalloc((void **)&dcounter, 4*sizeof(int));

  //copy the sequence from the vector into host array
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  //copy ha into da
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da
  //copy counter into device counter
  cudaMemcpy(dcounter, hcounter, 4*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  //call kernel function
  count<<<1,N>>>(da, dcounter, N);    

  cudaDeviceSynchronize();

  //copy device counter back into host counter
  cudaMemcpy(hcounter, dcounter, 4*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_results_file(hcounter, "output.txt");

  cudaFree(da);
  cudaFree(dcounter);

  free(ha);
  free(hcounter);

  return 0;
}