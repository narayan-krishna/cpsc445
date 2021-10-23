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

void read_str(vector<char> &str, string file_name){
    ifstream input_stream (file_name);
    char c;
    if(input_stream.is_open()){
      while(input_stream.get(c))
        str.push_back(c);    
    }
    input_stream.close();
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
  vector<char> sequence;
  int sequence_length;
  vector<char> cut;
  int results[4] = {0};
  vector<char> final_results;

  // cout << argv[1] << endl;

  read_str(sequence, "dna.txt");
  sequence_length = sequence.size();

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";

  int divisible = (sequence_length % p == 0 ? 0 : (p - sequence_length % p));

  // cout << (sequence_length + divisible)/p << endl; 
  cut.resize((sequence_length + divisible)/p);
  if (rank == 0) {
    final_results.resize(sequence_length + divisible);
    // cout << "final size" << final_results.size() << endl;
  }

  check_error(MPI_Scatter(&sequence[0], (sequence_length + divisible)/p, 
              MPI_CHAR, &cut[0], (sequence_length + divisible)/p, MPI_CHAR, 0, 
              MPI_COMM_WORLD));  

  invert_sequence(cut);

  check_error(MPI_Gather(&cut[0], 3, MPI_CHAR, &final_results[0],
              3, MPI_CHAR, 0, MPI_COMM_WORLD));
  // cout << rank << "sum: " << sum << endl;
  // sleep(1);
  if (rank==0) {
    print_vector(final_results); 
    print_vector_file(final_results, "output.txt");
    cout << endl;
  }

  // sleep(2);

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
