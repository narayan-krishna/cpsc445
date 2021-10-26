#include <mpi.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <unistd.h>
#include <map>
#include <fstream>

using namespace std;

/*DNA PARSE -- counts the number of TRIPLETS in a sequence*/

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

//print vector
void print_vector(vector<char> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

//read str from file into vec
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

/*maps each dna letter to a certain value so that letters and
  triplets can be altered using ints*/
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

//return a letter based of its index (reverse of previous function
char reverse_index(int num) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  return chars[num];
}

//given a the index of a certain triple (1 for triplet AAA, return the triplet
string reverse_translate(int triplet_index) {
  string combo;
  combo += reverse_index(triplet_index/16);
  triplet_index = triplet_index % 16;

  combo += reverse_index(triplet_index/4);
  triplet_index = triplet_index % 4;

  combo += reverse_index(triplet_index);

  return combo;
}

//counts triplets and adds counts to file, from vector of chars v
void count_triplets(const vector<char> &v, int *counter) {

  int length = v.size();
  int loc_store = 0; //stores the index of a triplet
  int skip = 0; //tracks whether triplet is somehow invalid and should be skip

  for(int i = 0; i < length; i+=3) { //for each triplet

    for(int j = 0; j < 3; j ++) { //for each char in triplet
      if(skip == 0) { //if don't skip
        int index_num = index(v[i + j]); 
        if (index_num > -1) {
          loc_store += index_num / pow(4, j); //add index num modified to loc
        } else {
          skip = 1; //invalid input? skip
        }
      }

    }

    if(skip == 0) { //wasn't skipped?
      counter[loc_store]++; //add to counter array
    } 

    loc_store = 0; //reset vals
    skip = 0;
  }
}

//print triplets in order (map) to stdout
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

//print triplets in order (map) to file
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
  //establish rank for process and total processes
  int rank;
  int p;

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  //info necessary to perform task on separate processes
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

    //calculate how to increase vector to make divisble
    //divisions also have to be divisible by 3 (triplets)
    sequence_length = sequence.size();
    divisible = ((p*3) - (sequence_length % (p*3)));
    cout << "divisible: " << divisible << endl;
    sequence.resize(sequence_length + divisible);
    sequence_length = sequence.size();

    cut_size = sequence_length/p;
  }

  //broadcast cut_size so that processes can resize to hold enough data
  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);

  //scatter data
  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  //cout << cut_size << endl;
  //count triplets in cuts
  count_triplets(cut, counts);

  //reduce counts to final results vector
  check_error(MPI_Reduce(&counts[0], &final_results[0], 64, MPI_INT, MPI_SUM, 
              0, MPI_COMM_WORLD));

  if (rank==0) {
    //print triplets to file
    print_triplets_file(final_results, "output.txt");
  }

  //finish
  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
