#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

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
  

void print_locs_file(const int *locs, int size, string file_name) {
  ofstream out_file;
  out_file.open(file_name, fstream::app); //open file

  bool within_sequence = false; //if already in a sequence, ignore begins
  int start = -1; //var to check start;
  for(int i = 0; i < size; i ++) {
    if(locs[i] == 1 && !within_sequence) { //if start + not in sequence
      start = i;
      within_sequence = true; //now in sequence

    } else if(locs[i] == 2 && within_sequence) {
      out_file << start << " " << i << endl;  //end found? print 
      within_sequence = false; //no longer in sequence
    }
  }
  out_file.close();
}


//the sequence da, sequence length, n
__global__ void locate(int *da, int *dlocs, int N) {
  int tid = threadIdx.x;
  int offset_loc = tid*3;

  printf("tid is: %i\n", tid);

  int seq_index = 0;
  seq_index += da[offset_loc] * 16;
  seq_index += da[offset_loc + 1] * 4;
  seq_index += da[offset_loc + 2] * 1;
  printf("%i, %i, %i\n", da[offset_loc], da[offset_loc + 1], da[offset_loc + 2]);

  if(seq_index == 6) {
    //update the location vector that this a potential start
    dlocs[tid] = 1; 
  } else if(seq_index == 16 || seq_index == 18 || seq_index == 24) {
    //update the location vector that this a potential end
    dlocs[tid] = 2;
  }

  //translate the number combination into number count
  // printf("loc_store is: %i\n", loc_store);

  // atomicAdd(&dlocs[seq_index], 1);
}

int main() {

  vector<int> temp_sequence;
  read_str(temp_sequence, "dna.txt");

  int N = temp_sequence.size();
  cout << N << endl;
  int divisible = N % 3;
  cout << "num elements to be added: " << 3 - divisible << endl;
  if (divisible != 0) { temp_sequence.resize(N + (3 - divisible)); }
  N = temp_sequence.size();
  cout << N << endl;

  int *ha = new int[N]; //the array
  int *hlocs = new int[N/3]{0}; //the location store for every third
  int *da, *dlocs; 

  cudaMalloc((void **)&da, N*sizeof(int));
  cudaMalloc((void **)&dlocs, N/3*sizeof(int));

  // set problem input (b)
  for (int i = 0; i<N; ++i) {
    ha[i] = temp_sequence[i];
  }
  puts("\n");
  
  cudaMemcpy(da, ha, N*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da
  cudaMemcpy(dlocs, hlocs, N/3*sizeof(int), cudaMemcpyHostToDevice); //copy ints from ha into da

  locate<<<1,N/3>>>(da, dlocs, N);    

  cudaDeviceSynchronize();

  cudaMemcpy(hlocs, dlocs, N/3*sizeof(int), cudaMemcpyDeviceToHost); //copy back value of da int sum

  for(int i = 0; i < N/3; i ++) {
    cout << hlocs[i];
  }
  cout << endl;

  print_locs_file(hlocs, N/3, "output.txt");

  cudaFree(da);
  cudaFree(dlocs);

  free(ha);
  free(hlocs);

  return 0;
}