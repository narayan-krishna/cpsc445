#include <iostream>
#include <algorithm>
#include <vector>

using namespace std;

int main () {

    int rows = 4;
    int columns = 4;
    int addr_1d = 10;

    vector<int> neighbors;

    neighbors.push_back(addr_1d - 1);
    neighbors.push_back(addr_1d + 1);
    neighbors.push_back(addr_1d - rows);
    neighbors.push_back(addr_1d - rows - 1);
    neighbors.push_back(addr_1d - rows + 1);
    neighbors.push_back(addr_1d + rows);
    neighbors.push_back(addr_1d + rows - 1);
    neighbors.push_back(addr_1d + rows + 1);

    sort(neighbors.begin(), neighbors.end());

    for(auto n : neighbors) {
        cout << n << endl;
    }

    return 0;
}