#include <mpi.h>
#include <iostream>
#include <string>
#include <math.h>
#include <vector>
#include <unistd.h>
#include <map>
#include <fstream>

using namespace std;

/*DNA GENES -- counts the number of As, Ts, Gs, and Cs in a sequence*/

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

//print a char vector
void print_vector(vector<char> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

//print a string vector
void print_vector(vector<int> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

//read a str from a file
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

/*given a the location of a sequence and it's index, determine
  if it could be the start or end of a gene*/
void is_start_or_end(const int seq_loc, const int seq_index, 
                     vector<int> &start_end_locs) {
  if(seq_index == 6) {
    //update the location vector that this a potential start
    start_end_locs[seq_loc] = 1;
  } else if(seq_index == 16 || seq_index == 18 || seq_index == 24) {
    //update the location vector that this a potential end
    start_end_locs[seq_loc] = 2;
  }
}

//search a given vector for starts or ends
void search_starts_ends(const vector<char> &v, vector<int> &start_end_locs) {
  // char chars[4] = {'A', 'T', 'G', 'C'}; 
  int length = v.size();
  int loc_store = 0; //stores the index of a triplet
  int skip = 0; //tracks whether triplet is somehow invalid and should be skip

  //iterate by 3
  for(int i = 0; i < length; i+=3) {

    //for each of the chars in a triplet
    for(int j = 0; j < 3; j ++) {
      //if don't skip
      if(skip == 0) {
        //calculate the index of num
        int index_num = index(v[i + j]);
        if (index_num > -1) {
          //properly reduce it while it valid
          loc_store += index_num / pow(4, j);
        } else {
          //if it's invalid, skip the whole triplet
          skip = 1;
        }
      }
    }

   if(skip == 0) {
      //if it wasn't skipped, check if it's a start or end
      //vector will be updated via reference
      is_start_or_end(i/3, loc_store, start_end_locs);
      // counter[loc_store]++;
    } 

    //reset
    loc_store = 0;
    skip = 0;
  }
}

//print locs vector to file
void print_locs_file(const vector<int> &locs, string file_name) {
  ofstream out_file;
  out_file.open(file_name, fstream::app); //open file

  bool within_sequence = false; //if already in a sequence, ignore begins
  int locs_size = locs.size();
  int start = -1; //var to check start;
  for(int i = 0; i < locs_size; i ++) {
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
  vector<int> start_end_locs;
  vector<int> final_results;

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
  start_end_locs.resize(cut_size/3);

  //scatter input string
  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  /*--------------------------------------------*/
  search_starts_ends(cut, start_end_locs);
  /*--------------------------------------------*/

  //resize final results to fit data
  if(rank == 0) {
    final_results.resize(start_end_locs.size() * p);
  }

  //gather results together
  check_error(MPI_Gather(&start_end_locs[0], start_end_locs.size(), MPI_INT, &final_results[0],
              start_end_locs.size(), MPI_INT, 0, MPI_COMM_WORLD));

  //print results to file
  if (rank==0) {
    print_locs_file(final_results, "output.txt");
  }

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
