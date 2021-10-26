#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <fstream>

using namespace std;

/*DNA INVERT -- inverts a sequence of DNA*/

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

//read a str from file into vector
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

//print vector
void print_vector(vector<char> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

//given a sequence, invert dna chars
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

//print a vector of dna chars to file
void print_vector_file(const vector<char> &v, string file_name) {
  ofstream out_file;
  out_file.open (file_name, fstream::app);
  for(char i : v) {
    out_file << i;
  }
  out_file << endl;
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
  vector<char> final_results;

  if(rank == 0) {
    read_str(sequence, "dna.txt");
    if (!sequence.size()) {
      cerr << "Invalid sequence length. Exiting..." << endl;
      exit(1);
    }
    //establish current length
    sequence_length = sequence.size();

    /*depending on string size, determine how to adjust sequence vector sizing 
      so that partitioning is even*/
    int divisible = (sequence_length % p == 0 ? 0 : (p - sequence_length % p));
    cut_size = (sequence_length + divisible)/p;
  }

  //broadcast the cut_size so all processes can adjust
  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);
  if (rank == 0) {
    //resize the results vector
    final_results.resize(sequence_length + divisible);
  }

  //scatter sequene to processes
  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  //invert cuts
  invert_sequence(cut);

  //gather cuts together
  check_error(MPI_Gather(&cut[0], cut_size, MPI_CHAR, &final_results[0],
              cut_size, MPI_CHAR, 0, MPI_COMM_WORLD));

  //resize and print final results vector
  if (rank==0) {
    final_results.resize(final_results.size() - divisible);
    print_vector_file(final_results, "output.txt");
  }


  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
