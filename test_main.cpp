#include <mpi.h>
#include <iostream>
#include <unistd.h>

using namespace std;

//run mpi while checking errors, take an error message
void check_error(int status, const string message="MPI error") {
  if ( status != 0 ) {    
    cerr << "Error: " << message << endl;
    exit(1);
  }
}

int main (int argc, char *argv[]) {
  int rank;
  int p;

  // Initialized MPI
  check_error(MPI_Init(&argc, &argv), "unable to initialize MPI");
  check_error(MPI_Comm_size(MPI_COMM_WORLD, &p), "unable to obtain p");
  check_error(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "unable to obtain rank");
  cout << "Starting process " << rank << "/" << "p\n";
  sleep(1);
  // example code
  // if the rank of this process is 0, n is 5. otherwise n is 0
  int n = 5, sum = 0;
  cout << rank << "n val: " << n << endl;
  sleep(1);

  //0 broadcasts n to tthe other processes
  //check_error(MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD));  
  //cout << rank << ": " << n << endl;

  //sum all values n
  check_error(MPI_Reduce(&n, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  sleep(1);
  cout << rank << "sum: " << sum << endl;
  if (rank==0) {
    // for the 0 rank, return an error if sum != 0 * processes
    if (sum != n*p) { cerr << "error!\n"; exit(1); }
  }

  check_error(MPI_Finalize());

  cout << "Ending process " << rank << "/" << "p\n";

  return 0;
}
