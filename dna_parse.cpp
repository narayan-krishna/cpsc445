#include <mpi.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <unistd.h>
#include <map>
#include <fstream>

using namespace std;

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

void print_vector(vector<char> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

int read_str(vector<char> &str, string file_name){
    ifstream input_stream (file_name);
    char c;
    if(input_stream.is_open()){
      while(input_stream.get(c))
        str.push_back(c);    
    }
    input_stream.close();
    return 0;
}

int index(const char &letter) {
  if(letter == 'A') {
    return 0;
  } else if(letter == 'T') {
    return 16;
  } else if(letter == 'G') {
    return 32;
  } else if(letter == 'C') {
    return 48;
  }
  return -1;
}

char reverse_index(int num) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  return chars[num];
}

string reverse_translate(int triplet_index) {
  string combo;
  combo += reverse_index(triplet_index/16);
  triplet_index = triplet_index % 16;

  combo += reverse_index(triplet_index/4);
  triplet_index = triplet_index % 4;

  combo += reverse_index(triplet_index);

  return combo;
}

void count_triplets(const vector<char> &v, int *counter) {
  // char chars[4] = {'A', 'T', 'G', 'C'}; 
  int length = v.size();
  int loc_store = 0;
  int skip = 0;
  for(int i = 0; i < length; i+=3) {

    for(int j = 0; j < 3; j ++) {
      if(skip == 0) {
        int index_num = index(v[i + j]);
        if (index_num > -1) {
          loc_store += index_num / pow(4, j);
        } else {
          skip = 1;
        }
      }

    }

    if(skip == 0) {
      counter[loc_store]++;
    } 

    loc_store = 0;
    skip = 0;
  }
}

void print_triplets_stdout(const int *triplet_counter) {
  map<string, int> results;
  for(int i = 0; i < 64; i ++) {
    if(triplet_counter[i] != 0) {
      // cout << reverse_translate(i) << " " << triplet_counter[i] << endl;
      results[reverse_translate(i)] = triplet_counter[i];
    }
  }

  for(auto k : results) {
    cout << k.first << " " << k.second << endl;
  }
  cout << endl;
}

void print_triplets_file(const int *triplet_counter, string file_name) {
  map<string, int> results;
  for(int i = 0; i < 64; i ++) {
    if(triplet_counter[i] != 0) {
      // cout << reverse_translate(i) << " " << triplet_counter[i] << endl;
      results[reverse_translate(i)] = triplet_counter[i];
    }
  }

  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(auto k : results) {
    out_file << k.first << " " << k.second << endl;
  }
  out_file.close();
}

int main (int argc, char *argv[]) {
  int rank;
  int p;

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  vector<char> sequence;
  int sequence_length;
  int cut_size;
  int divisible;
  vector<char> cut;
  // vector<char> final_results; 
  int counts[64] = {0};
  int final_results[64] = {0};
  // char **sequences;

  if(rank == 0) {
    read_str(sequence, "dna.txt");
    if (!sequence.size()) {
      cerr << "Invalid sequence length. Exiting..." << endl;
      exit(1);
    }

    sequence_length = sequence.size();
    // if(sequence_length % p*3 != 0) {
    //   // cout << "no" << endl;
    // } else {
    //   int current_cut = sequence_length/p;
    //   if(current_cut%3 != 0) {
    //     divisible = (current_cut - (current_cut % 3))/3;
    //     cout << "divisible: " << divisible << endl;
    //     // cout << "cut size maybe: " << (sequence_length + divisible)/p << endl;
    //   } else {

    //     divisible = 0;
    //     cout << "yes" << endl;
    //   }
    // }
    divisible = ((p*3) - (sequence_length % (p*3)));
    cout << "divisible: " << divisible << endl;
    sequence.resize(sequence_length + divisible);
    sequence_length = sequence.size();

    cut_size = sequence_length/p;
  }

  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);

  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  cout << cut_size << endl;
  count_triplets(cut, counts);

  check_error(MPI_Reduce(&counts[0], &final_results[0], 64, MPI_INT, MPI_SUM, 
              0, MPI_COMM_WORLD));

  if (rank==0) {
    print_triplets_file(final_results, "output.txt");
  }

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
