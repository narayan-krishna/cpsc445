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

void print_vector(vector<int> &vec) {
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
}

void is_start_or_end(const int seq_loc, const int seq_index, 
                     vector<int> &start_end_locs) {
  if(seq_index == 6) {
    start_end_locs[seq_loc] = 1;
  } else if(seq_index == 16 || seq_index == 18 || seq_index == 24) {
    start_end_locs[seq_loc] = 2;
  }
}

void search_starts_ends(const vector<char> &v, vector<int> &start_end_locs) {
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
      is_start_or_end(i/3, loc_store, start_end_locs);
      // counter[loc_store]++;
    } 

    loc_store = 0;
    skip = 0;
  }
}

void count_triplets(const vector<char> &v, int *counter, vector<int> &start_end_locs) {
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
      // cout << loc_store << endl;
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

void print_locs_file(const vector<int> &locs, string file_name) {
  ofstream out_file;
  out_file.open(file_name, fstream::app);

  bool within_sequence = false;
  int locs_size = locs.size();
  int start = -1; //int end = -1;
  for(int i = 0; i < locs_size; i ++) {
    // cout << locs[i];
    if(locs[i] == 1 && !within_sequence) {
      start = i;
      within_sequence = true;

    } else if(locs[i] == 2 && within_sequence) {
      out_file << start << " " << i << endl;   
      within_sequence = false;
    }
  }
  out_file.close();
}

// void delete_2d_array(char **array, int array_size) {
// }

int main (int argc, char *argv[]) {
  cout << "hello world" << endl;
  // int rank;
  // int p;

  // // Initialized MPI
  // check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  // check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  // check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  // cout << "Starting process " << rank << "/" << "p\n";

  // vector<char> sequence;
  // int sequence_length;
  // int cut_size;
  // int divisible;
  // vector<char> cut;
  // vector<int> start_end_locs;
  // // vector<char> final_results; 
  // // int counts[64] = {0};
  // // int final_results[64] = {0};
  // // char **sequences;
  // vector<int> final_results;

  // if(rank == 0) {
  //   read_str(sequence, "dna.txt");
  //   if (!sequence.size()) {
  //     cerr << "Invalid sequence length. Exiting..." << endl;
  //     exit(1);
  //   }
  //   sequence_length = sequence.size();
  //   divisible = ((p*3) - (sequence_length % (p*3)));
  //   cout << "divisible: " << divisible << endl;
  //   sequence.resize(sequence_length + divisible);
  //   sequence_length = sequence.size();

  //   cut_size = sequence_length/p;
  // }

  // check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  // cut.resize(cut_size);
  // start_end_locs.resize(cut_size/3);

  // check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
  //                         MPI_CHAR, 0, MPI_COMM_WORLD));  

  // /*--------------------------------------------*/
  // search_starts_ends(cut, start_end_locs);
  // // count_triplets(cut, counts, start_end_locs);
  // /*--------------------------------------------*/

  // // sleep(.2);
  // // cout << rank << ": "; print_vector(cut); cout << endl;
  // // sleep(1);
  // // cout << rank << ": "; print_vector(start_end_locs); cout << endl;
  // // sleep(1);

  // if(rank == 0) {
  //   final_results.resize(start_end_locs.size() * p);
  // }

  // // check_error(MPI_Reduce(&counts[0], &final_results[0], 64, MPI_INT, MPI_SUM, 
  // //             0, MPI_COMM_WORLD));

  // check_error(MPI_Gather(&start_end_locs[0], start_end_locs.size(), MPI_INT, &final_results[0],
  //             start_end_locs.size(), MPI_INT, 0, MPI_COMM_WORLD));

  // if (rank==0) {
  //   print_vector(final_results); cout << endl;
  //   print_locs_file(final_results, "output.txt");
  //   // char combo[3] = {'A','T','G'};
  //   // cout << index(&combo) << endl;
  //   // cout << index() << endl;
  //   // cout << index() << endl;
  //   // cout << index() << endl;
  //   // print_triplets_file(final_results, "output.txt");
  // }

  // check_error(MPI_Finalize());
  // cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
