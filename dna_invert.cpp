#include <mpi.h>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>
#include <fstream>

using namespace std;

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
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

void print_vector(vector<char> &vec) {
  for(auto i : vec) {
    cout << i;
  }
}

void calculate_partition_range(int &start, int &end, const int &size,
                               const int &p, const int &r) { 
  int task_size = (size/p) + ((r < size%p) ? 1 : 0);
  start = r*(size/p) + min(r, size%p);
  end = start + task_size;
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

void print_results_stdout(const int *char_counter) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  for(int i = 0; i < 4; i ++) {
    cout << chars[i] << " " << char_counter[i] << endl;
  }
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
  vector<char> final_results;

  if(rank == 0) {
    read_str(sequence, "dna.txt");
    if (!sequence.size()) {
      cerr << "Invalid sequence length. Exiting..." << endl;
      exit(1);
    }
    sequence_length = sequence.size();
    int divisible = (sequence_length % p == 0 ? 0 : (p - sequence_length % p));
    cut_size = (sequence_length + divisible)/p;
  }

  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);
  if (rank == 0) {
    final_results.resize(sequence_length + divisible);
    // cout << "final size" << final_results.size() << endl;
  }

  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  invert_sequence(cut);

  check_error(MPI_Gather(&cut[0], cut_size, MPI_CHAR, &final_results[0],
              cut_size, MPI_CHAR, 0, MPI_COMM_WORLD));
  // cout << rank << "sum: " << sum << endl;
  // sleep(1);
  if (rank==0) {
    final_results.resize(final_results.size() - divisible);
    // print_vector(final_results); 
    print_vector_file(final_results, "output.txt");
    // cout << endl;
  }

  // sleep(2);

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
