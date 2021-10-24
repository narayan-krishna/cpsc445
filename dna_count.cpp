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
  vector<char> cut;
  int results[4] = {0};
  int final_results[4] = {0};

  if(rank == 0) {
    read_str(sequence, "dna.txt");
    if (!sequence.size()) {
      return 0;
    }
    sequence_length = sequence.size();
    int divisible = (sequence_length % p == 0 ? 0 : (p - sequence_length % p));
    cut_size = (sequence_length + divisible)/p;
  }

  check_error(MPI_Bcast(&cut_size, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  cut.resize(cut_size);

  check_error(MPI_Scatter(&sequence[0], cut_size, MPI_CHAR, &cut[0], cut_size, 
                          MPI_CHAR, 0, MPI_COMM_WORLD));  

  count_sequence(results, cut);

  //barrier here
  check_error(MPI_Reduce(&results[0], &final_results[0], 4, MPI_INT, MPI_SUM, 
              0, MPI_COMM_WORLD));
  // cout << rank << "sum: " << sum << endl;
  if (rank==0) {
    print_results_file(final_results, "output.txt");
  }

  // sleep(2);

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}  
