#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <math.h>

using namespace std;

//read string from file into a vector -> translate chars to ints
void read_csv(vector<float> &values, const string &path, int &column_count){
    ifstream input_stream (path);

    if (!input_stream.is_open()) {
      cerr << "coudn't find/open file..." << endl;
      exit(EXIT_FAILURE);
    }

    //how many columns are there?

    for(string line; getline(input_stream, line); column_count ++) {
      cout << line << endl;
      stringstream ss(line);

      string float_string;
      while(getline(ss, float_string, ',')) {
        values.push_back( (float)atof(float_string.c_str()) );
      }
    }
}

inline void get_resident_coords(const int &index, int &x_coord, int &y_coord, int &rows) {
    x_coord = (index / rows);
    y_coord = (index % rows); 
}

//print a sequence of characters to a file
void print_to_csv(const bool *sequence, int length, int rows, string output_file) {

  ofstream out_file;
  out_file.open (output_file);

  for(int i = 0; i < length; i ++) {
    // if (sequence[i] == 1) {
    //   int x_coord; int y_coord;
    //   get_resident_coords(i, x_coord, y_coord, rows);
    //   out_file << x_coord << ", " << y_coord << endl; 
    // }
    cout << i << ", " << sequence[i] << endl;
  }

  out_file.close();
  // system("head output.csv");
  // system("rm output.csv");
}

__device__
bool is_smaller_or_greater(float *da, const int &addr_1d, const int &rows, const int &N) {
  // cout << here << endl;
  bool check_for_smaller = false;
  bool decided = false;

  int neighbors[8]; //eight surrounding neighbors
  neighbors[0] = addr_1d - 1;
  neighbors[1] = addr_1d + 1;

  neighbors[2] = addr_1d - rows;

  neighbors[3] = addr_1d - rows - 1;
  neighbors[4] = addr_1d - rows + 1;

  neighbors[5] = addr_1d + rows;

  neighbors[6] = addr_1d + rows - 1;
  neighbors[7] = addr_1d + rows + 1;

  int neighbor_sum = 0;
  for(int i = 0; i < 8; i ++) {
    if(neighbors[i] > 0 && neighbors[i] < N) { //ignore if nieghbor is negative/outofgrid
      //is the nieghbor smaller than the current cell?
      // bool is_smaller = (da[neighbors[i]] < da[addr_1d]);
      // if (decided) { //if we already know we're looking for g/s
      //   if (is_smaller != check_for_smaller) { //if we dont' match the condition 
      //                                          //we're checking for
      //     return false; //return false
      //   }
      // } else { //if we haven't decided, decided will be this 
      //   check_for_smaller = is_smaller;
      //   decided = true;
      // }
      neighbor_sum ++;
    }
  }
  // printf("\n");
  if (neighbor_sum == 8) return true;
  return false;

}

__global__ 
void extreme(float *da, bool *dbools, int N, int rows, int columns) {

  // int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // __shared__ float s[512];
  //allocated 512 floats per block
  //copy over 512 floats into the corresponding block

  // s[tid] = da[gid]; //copy everything
  // __syncthreads();
  dbools[gid] = is_smaller_or_greater(da, gid, rows, N);
  // printf("tid is: %i, seeing value: %f\n", tid, da[tid]);
}

/**
 * this program will
 * take an array of elements
 * it will check its neighbors
 * */

int main() {
  cout << "\ncsv input head --------------------" << endl;
  system("head input.csv");
  cout << "\n-----------------------------" << endl;

  //INPUTS
  int N = 0;
  int rows = 0; int columns = 0;

  //read into vector so for dynamic length + size checking
  vector<float> inputs;
  read_csv(inputs, "input.csv", columns);
  cout << "\ninputs(" << inputs.size() << ")" << endl;

  N = inputs.size();

  rows = N/columns;
  printf("rows: %i | columns: %i\n", rows, columns);

  vector<vector<float>> inputs_2d;
  for(int i = 0; i < rows + 2; i ++) {
    for(int j = 0; j < columns + 2; j ++) {
      inputs_2d[i][j] = 0;
    }
  }

  for(int i = 1; i < rows + 1; i ++) {
    for(int j = 1; j < columns + 1; j ++) {
      inputs_2d[i][j] = inputs[(i*j) - 1];
    }
  }
  
  for(int i = 0; i < rows + 2; i ++) {
    for(int j = 0; j < columns + 2; j ++) {
      cout << inputs_2d[i][j];
    }
    cout << endl;
  }

  float *ha = new float[N];
  bool *hbools = new bool[N]();

  for(int i = 0; i < 20; i ++) {
    cout << inputs[i] << ", ";
  }
  cout << "..." << endl; cout << endl;

  float *da; bool *dbools;
  cudaMalloc((void **) &da, N*sizeof(float));
  cudaMalloc((void **) &dbools, N*sizeof(bool));
  cudaMemcpy(da, ha, N*sizeof(float), cudaMemcpyHostToDevice); //copy ints from ha into da
  cudaMemcpy(dbools, hbools, N*sizeof(bool), cudaMemcpyHostToDevice); //copy ints from ha into da

  int Nthreads = 512;
  int Nblocks = (N + (Nthreads - 1)) / Nthreads;
  cout << Nthreads << ", " << Nblocks << endl;
  extreme<<<Nblocks,Nthreads>>>(da, dbools, N, rows, columns);
  cudaDeviceSynchronize();

  cudaMemcpy(ha, da, N*sizeof(float), cudaMemcpyDeviceToHost); //copy back value of da int sum
  cudaMemcpy(hbools, dbools, N*sizeof(bool), cudaMemcpyDeviceToHost); //copy back value of da int sum

  print_to_csv(hbools, N, rows, "output.csv");

  cout << "head output csv" << "--------------" << endl;
  system("cat output.csv");
  cout << "------------------------------------" << endl;

  cudaFree(da);
  free(ha);
  return 0;
}