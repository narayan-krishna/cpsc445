#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

using namespace std;

//read sequence string into vector from file
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

//reverse translate a triplet index into it's corresponding index
//using base 4 math 
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
  
//order all results in a map and then print them out
void print_results_file(const int *combo_counter, string file_name) {
  map<string, int> results;
  for(int i = 0; i < 64; i ++) {
    if(combo_counter[i] != 0) {
      // cout << reverse_translate(i) << " " << triplet_counter[i] << endl;
      results[reverse_translate(i)] = combo_counter[i];
    }
  }

  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(auto k : results) {
    out_file << k.first << " " << k.second << endl;
  }
  out_file.close();
}

//the sequence da, sequence length, n
//parse the sequence for triplets, count them
__global__ void parse(int *da, int *dcounter, int N) {
  int tid = threadIdx.x;
  int offset_loc = tid*3;

  printf("tid is: %i\n", tid);

  //calculate index of triplet from 0 to 63, add it to the counter
  int loc_store = 0;
  loc_store += da[offset_loc] * 16;
  loc_store += da[offset_loc + 1] * 4;
  loc_store += da[offset_loc + 2] * 1;
  printf("%i, %i, %i\n", da[offset_loc], da[offset_loc + 1], da[offset_loc + 2]);

  //translate the number combination into number count
  printf("loc_store is: %i\n", loc_store);

  atomicAdd(&dcounter[loc_store], 1);
}

int main() {

  //vector the acquire sequence from dna.txt
  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  //ensure that sequence length is divisible by 3 so that triplets can be counted

  //get the size of the sequence
  int N = temp_sequence.size();
  cout << N << endl;
  int divisible = N % 3; //if its divisible then this should be 0
  cout << "num elements to be added: " << 3 - divisible << endl;
  //otherwise, compensate using calculation
  if (divisible != 0) { temp_sequence.resize(N + (3 - divisible)); }
  //reacquire size for future partitioning purposes
  N = temp_sequence.size();
  cout << N << endl;

  //allocate arrays
  int *ha = new int[N]; //host array for sequence
  int *hcounter = new int[64]{0}; //host counter for all 64 variations of
                                  //triplets
  int *da, *dcounter; //declare device array, device counter

  cudaMalloc((void **)&da, N*sizeof(int)); //allocate to gpu memory
  cudaMalloc((void **)&dcounter, 64*sizeof(int)); //allocate to gpu memory

  //copy vector into host array
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  //copy host array into device array, copy host counter into device counter
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da
  cudaMemcpy(dcounter, hcounter, 64*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  //call kernel function
  parse<<<1,N/3>>>(da, dcounter, N);    

  cudaDeviceSynchronize();

  //copy device counter back into host counter
  cudaMemcpy(hcounter, dcounter, 64*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  //print the results back to file
  print_results_file(hcounter, "output.txt");

  //free allocated memory
  cudaFree(da);
  cudaFree(dcounter);

  free(ha);
  free(hcounter);

  return 0;
}