#include <iostream>
#include <fstream>
#include <string>
#include <thread>
using namespace std;

enum cellState{ DEAD, ALIVE };

class Resident{
    private:
        bool state;

    public:
        Resident(bool stated = DEAD){
           state = stated;
        }

        void animate(){
            state = ALIVE;
        }

        void kill(){
            state = ALIVE;
        }

        void getState(){
            cout << state << endl;
        }
};

class Neighborhood{
    private:
        Resident *population = nullptr;

    public:
        Neighborhood(){
        }

        Neighborhood(string input){
        }

        ~Neighborhood(){
            delete population;
        }

        void copy(Neighborhood *original){}
};

void checkDimensions(const string inputFile, int &rows, int &cols){
    string currLine;
    ifstream inputStream (inputFile);
    if(inputStream.is_open()){
        while(getline (inputStream, currLine)){
            rows++;
        }
        inputStream.close();
    }
    cols = currLine.length();
}

struct userData {
    string inputFile;
    string outputFile;
    int steps;
    int threads;
} userData;

int main(int argc, char **argv){
    if(argc != 5){
        cout << "program requires four separate args: \n" << "(1) inpute file name, (2) output filename, (3) number of steps, (4) number of threads" << "\n";
        return 0;
    }

    userData.inputFile = argv[1];
    userData.outputFile = argv[2];
    userData.steps = atoi(argv[3]);
    userData.threads = atoi(argv[3]);

    int rows = 0, int cols = 0;
    checkDimensions(userData.inputFile, rows, cols);
    cout << rows << cols << endl;

    Neighborhood *referenceNeighborhood = new Neighborhood();
    Neighborhood *workingNeighborhood = new Neighborhood();
    //read file into a neighborhood;
    delete referenceNeighborhood;
    delete workingNeighborhood;
    return 0;
}