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

void read_str(vector<char> &str){
    ifstream input_stream ("dna.txt");
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

void count_sequence(int *results, const vector<char> &sequence, 
                    const int &start, const int &end) {
  for(int i = start; i < end; i ++) {
    char curr = sequence[i];
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

void print_results_file(const int *char_counter) {
  char chars[4] = {'A', 'T', 'G', 'C'}; 
  ofstream out_file;
  out_file.open ("output.txt", fstream::app);
  for(int i = 0; i < 4; i ++) {
    out_file << chars[i] << " " << char_counter[i] << endl;
  }
  out_file.close();
}

int main (int argc, char *argv[]) {
  int rank;
  int p;
  vector<char> sequence;
  int sequence_length;
  int results[4] = {0};
  int final_results[4] = {0};

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";
  
  if(rank == 0) {
    read_str(sequence);
    // print_vector(sequence); cout << endl;
    sequence_length = sequence.size();
  }
  
  /*partition*/
  //sleep(1);
  // int n = (rank==0?5:0), sum = 0;
  // cout << rank << "n val: " << n << endl;
  //sleep(1);

  //0 broadcasts n to the other processes
  check_error(MPI_Bcast(&sequence_length, 1, MPI_INT, 0, 
              MPI_COMM_WORLD));  


  if(rank != 0) {
    sequence.resize(sequence_length);
  }

  check_error(MPI_Bcast(&sequence[0], sequence_length, MPI_CHAR, 0, 
              MPI_COMM_WORLD));  

  int range_start; int range_end;
  calculate_partition_range(range_start, range_end, sequence_length, p, rank);

  // sleep(1);
  // cout << "rank " << rank << ": " << range_start << ", " << range_end << endl;

  // sleep(1);
  // cout << rank << "str: ";
  // print_vector(sequence); cout << endl;
 

  //reduce all n values to a sum in root rank buffer
  // sleep(1);
  count_sequence(results, sequence, range_start, range_end);

  //barrier here
  check_error(MPI_Reduce(&results[0], &final_results[0], 4, MPI_INT, MPI_SUM, 
              0, MPI_COMM_WORLD));
  //cout << rank << "sum: " << sum << endl;
  if (rank==0) {
    // cout << range_start << ", " << range_end << endl;
    print_results_file(final_results);
  }

  check_error(MPI_Finalize());
  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
