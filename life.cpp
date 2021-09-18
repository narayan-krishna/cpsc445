#include <iostream>
#include <fstream>
#include <string>
#include <thread>
using namespace std;

enum cellState{ DEAD, ALIVE };

struct Dimensions {
    int rows = 0;
    int cols = 0;
} Dimensions;

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

        bool getState(){
            return state;
        }

        friend ostream& operator<< (ostream& os, const Resident& resident){
            os << resident.state;
            return os;
        }

};

class Neighborhood{
    private:
        Resident **residents { nullptr };

    public:
        Neighborhood(){
            residents = new Resident *[Dimensions.rows];
            for(int i = 0; i < Dimensions.rows; ++i){
                residents[i] = new Resident[Dimensions.cols];
            }
        }

        Neighborhood(string input){
        }

        ~Neighborhood(){
            for(int i = 0; i < Dimensions.rows; ++i){
                delete[] residents[i];
            }
            delete[] residents;
        }

        void copy(Neighborhood *original){}

        void print(){
            if(residents){
                for(int r = 0; r < Dimensions.rows; ++r){
                    for(int c = 0; c < Dimensions.cols; ++c){
                        cout << residents[r][c]; 
                    }
                    cout << endl;
                }
            }
        }
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

struct InitData {
    string inputFile;
    string outputFile;
    int steps;
    int threads;
} InitData;


int main(int argc, char **argv){

    if(argc != 5){
        cout << "program requires four separate args: \n" << "(1) inpute file name, (2) output filename, (3) number of steps, (4) number of threads" << "\n";
        return 0;
    }

    InitData.inputFile = argv[1];
    InitData.outputFile = argv[2];
    InitData.steps = atoi(argv[3]);
    InitData.threads = atoi(argv[4]);
    checkDimensions(InitData.inputFile, Dimensions.rows, Dimensions.cols);
    cout << Dimensions.rows << Dimensions.cols << endl;

    Neighborhood *referenceNeighborhood = new Neighborhood();
    //update the referenceNeighborhood using the input
    referenceNeighborhood->print();
    // Neighborhood *workingNeighborhood = new Neighborhood();
    //read file into a neighborhood;
    delete referenceNeighborhood;
    // delete workingNeighborhood;
    return 0;
}