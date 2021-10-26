#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <fstream>

using namespace std;

/*DNA COUNT -- counts the number of As, Ts, Gs, and Cs in a sequence*/

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}


//read a string from a file into a vector
void read_str(vector<char> &str, string file_name){
    ifstream input_stream (file_name);
    char c;
    if(input_stream.is_open()){
      while(input_stream.get(c))
        str.push_back(c);    
    }
    input_stream.close();
}

//print a character vector
void print_vector(vector<char> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

//add letter counts to a result array given a sequence
void count_sequence(int *results, const vector<char> &sequence) {
  for(auto curr : sequence) {
    // cout << curr << endl;
    if(curr == 'A') {
      results[0] ++;
    } else if(curr == 'T') {
      results[1] ++;
    } else if(curr == 'G') {
      results[2] ++;
    } else if(curr == 'C') {
      results[3] ++;
    }
  }
}

//print results array to standard output
void print_results_stdout(const int *char_counter) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  for(int i = 0; i < 4; i ++) {
    cout << chars[i] << " " << char_counter[i] << endl;
  }
}

//print results array to file 
void print_results_file(const int *char_counter, string file_name) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(int i = 0; i < 4; i ++) {
    out_file << chars[i] << " " << char_counter[i] << endl;
  }
  out_file.close();
}

int main (int argc, char *argv[]) {
  //initialize ranks and amount of processes 
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
  vector<char> cut;
  int results[4] = {0};
  int final_results[4] = {0};

  if(rank == 0) {
    //rank 0 handles input of string
    read_str(sequence, "dna.txt");
    if (!sequence.size()) {
      return 0;
    }
    /*depending on string size, determine how to adjust sequence vector sizing 
      so that partitioning is even*/

    //establish current length
    sequence_length = sequence.size();
    //calculate amount that vector needs to increase to make divisible
    int divisible = (sequence_length % p == 0 ? 0 : (p - sequence_length % p));
    //establish this as size of individual cuts or partitions per process
    cut_size = (sequence_length + divisible)/p;
  }

  //broadcast cut_size so that processes can resize to hold enough data
  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);

  //scatter input string
  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  //count cut sequences and add to each result array
  count_sequence(results, cut);

  //sum results using mpi reduce to an array final results
  check_error(MPI_Reduce(&results[0], &final_results[0], 4, MPI_INT, MPI_SUM, 
              0, MPI_COMM_WORLD));

  //print results from rank 0
  if (rank==0) {
    print_results_file(final_results, "output.txt");
  }


  //finalize and quit mpi, ending processes
  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
