#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

bool is_smaller_or_greater(float *da, const int &addr_1d, const int &rows, const int &N) {
  // cout << here << endl;
  bool check_for_smaller;
  bool decided = false;

  int neighbors[8]; //eight surrounding neighbors
  neighbors[0] = addr_1d - 1;
  neighbors[1] = addr_1d + 1;
  neighbors[2] = addr_1d - rows;
  neighbors[3] = addr_1d - rows - 1;
  neighbors[4] = addr_1d - rows + 1;
  neighbors[5] = addr_1d + rows;
  neighbors[6] = addr_1d + rows - 1;
  neighbors[7] = addr_1d + rows + 1;

  printf("id %i ---", addr_1d);
  for(int i = 0; i < 8; i ++) {
    if(neighbors[i] > 0 && neighbors[i] < N) { //ignore if nieghbor is negative/outofgrid
      //is the nieghbor smaller than the current cell?
      bool is_smaller = (da[neighbors[i]] < da[addr_1d]);
      if (decided) { //if we already know we're looking for g/s
        if (is_smaller != check_for_smaller) { //if we dont' match the condition 
                                               //we're checking for
          return false; //return false
        }
      } else { //if we haven't decided, decided will be this 
        check_for_smaller = is_smaller;
        decided = true;
      }
    }
    printf("%i,", neighbors[i]);
  }
  printf("\n");

  return true;
}

int main () {

    int rows = 4;
    int columns = 4;
    int addr_1d = 10;

    vector<int> neighbors;

    sort(neighbors.begin(), neighbors.end());

    for(auto n : neighbors) {
        cout << n << endl;
    }

    return 0;
}