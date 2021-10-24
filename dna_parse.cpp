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

int read_str_triplets(vector<vector<char>> &container, string file_name){
    ifstream input_stream (file_name);
    char c;
    if(input_stream.is_open()){

      int count = 0;
      vector<char> v;

      while(input_stream.get(c)) {
        if(count == 3) {
          container.push_back(v);
          v.clear();
          count = 0;
        }
        v.push_back(c);
        // print_vector(v); cout << endl;
        count ++;
      }
    }
    input_stream.close();
    return 0;
}

void invert_sequence(vector<char> &sequence) {
  for(int i = 0; i < sequence.size(); i ++) {
    char curr = sequence[i];
    if(curr == 'A') {
      sequence[i] = 'T';
    } else if(curr == 'T') {
      sequence[i] = 'A';
    } else if(curr == 'G') {
      sequence[i] = 'C';
    } else if(curr == 'C') {
      sequence[i] = 'G';
    }
  }
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
  // triplex_index -= triplet_index;
  //triplet_index -= triplet_index / 16;
  // cout << "triple index 1: " << triplet_index << endl;

  // if(triplet_index < 16) {
  //   combo += 'A';
  // } else if(triplet_index < 32) {
  //   combo += 'T';
  //   triplet_index -= 16;
  // } else if(triplet_index < 48) {
  //   combo += 'G';
  //   triplet_index -= 32;
  // } else {
  //   combo += 'C';
  //   triplet_index -= 48;
  // }
  
  // cout << "triple index: " << triplet_index << endl;

  // if(triplet_index < 4) {
  //   combo += 'A';
  // } else if(triplet_index < 8) {
  //   combo += 'T';
  //   triplet_index -= 4;
  // } else if(triplet_index < 12) {
  //   combo += 'G';
  //   triplet_index -= 8;
  // } else {
  //   combo += 'C';
  //   triplet_index -= 12;
  // }

  // cout << "triple index: " << triplet_index << endl;

  // if(triplet_index < 1) {
  //   combo += 'A';
  // } else if(triplet_index < 2) {
  //   combo += 'T';
  // } else if(triplet_index < 3) {
  //   combo += 'G';
  // } else {
  //   combo += 'C';
  // }

  cout << combo << endl;
}

void count_triplets(const vector<char> &v, int *counter) {
  // char chars[4] = {'A', 'T', 'G', 'C'}; 
  int length = v.size();
  int loc_store = 0;
  int skip = 0;
  for(int i = 0; i < length; i+=3) {
    // char first_char = v[i]; 
    // char second_char = v[i+1]; 
    // char third_char = v[i+2]; 
    // loc_store = index(first_char)/1 +
    //             index(second_char)/4 +
    //             index(third_char)/16;
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
    // cout << index(first_char)/1 << index(second_char)/4 << index(third_char)/16 << ": ";
    // cout << first_char << second_char << third_char << ": ";
    // cout << "index calc: " << loc_store << endl;
    if(skip == 0) {
      counter[loc_store]++;
    } 

    loc_store = 0;
    skip = 0;
  }
}


void print_results_stdout(const int *char_counter) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  for(int i = 0; i < 4; i ++) {
    cout << chars[i] << " " << char_counter[i] << endl;
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

void print_results_file(const int *char_counter, string file_name) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(int i = 0; i < 4; i ++) {
    out_file << chars[i] << " " << char_counter[i] << endl;
  }
  out_file.close();
}

void print_vector_file(const vector<char> &v, string file_name) {
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(char i : v) {
    out_file << i;
  }
  out_file << endl;
  out_file.close();
}

// void delete_2d_array(char **array, int array_size) {
// }

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
    if(sequence_length % p*3 != 0) {
      divisible = ((p*3) - (sequence_length % (p*3)));
      cout << "divisible: " << divisible << endl;
      sequence.resize(sequence_length + divisible);
      sequence_length = sequence.size();
      // cout << "no" << endl;
    } else {
      divisible = 0;
      cout << "yes" << endl;
    }
    // (add sequence_length % p*3 - p) to vector size
    // int divisible = (sequence_length % p == 0 ? 0 : (p - sequence_length % p));
    cut_size = sequence_length/p;
    // for(auto n : sequences) {
    //   cout << "[";
    //   for(auto i : n) {
    //     cout << i;
    //   }
    //   cout << "]" << endl;
    // }
  }

  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);
  // if (rank == 0) {
    // final_results.resize(sequence_length);
    // cout << "final size" << final_results.size() << endl;
  // }

  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  sleep(.2);
  cout << rank << ": "; print_vector(cut); cout << endl;

  // sleep(.2);
  // if(rank == 0) {
  //   // reverse_translate(0);
  //   // reverse_translate(16);
  //   // reverse_translate(32);
  //   // reverse_translate(64);
  //   for(int i = 0; i < 64; i ++) {
  //     if (i%8 == 0) cout << endl;
  //     cout << counts[i];
  //   }
  //   cout << endl;
  // }

  count_triplets(cut, counts);

  // check_error(MPI_Gather(&cut[0], cut_size, MPI_CHAR, &final_results[0],
  //             cut_size, MPI_CHAR, 0, MPI_COMM_WORLD));
  // cout << rank << "sum: " << sum << endl;
  // sleep(1);
  check_error(MPI_Reduce(&counts[0], &final_results[0], 64, MPI_INT, MPI_SUM, 
              0, MPI_COMM_WORLD));

  if (rank==0) {
    // for(int i = 0; i < 64; i ++) {
    //   if (i%8 == 0) cout << endl;
    //   cout << final_results[i];
    // }
    // cout << endl;
    print_triplets_file(final_results, "output.txt");
    // final_results.resize(sequence_length - divisible);
    // print_vector(final_results); 
    // print_vector_file(final_results, "output.txt");
    // cout << endl;
  }

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
