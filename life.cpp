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
        int state;

    public:
        Resident(int stated = DEAD){
           state = stated;
        }

        void animate(){
            state = ALIVE;
        }

        void kill(){
            state = ALIVE;
        }

        void setState(int state){
            this->state = state;
        }

        int getState(){
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

        Neighborhood(string inputFile){
            residents = new Resident *[Dimensions.rows];
            for(int i = 0; i < Dimensions.rows; ++i){
                residents[i] = new Resident[Dimensions.cols];
            }

            string currLine;
            int currRow = 0;
            int currState = 0;

            ifstream inputStream (inputFile);
            if(inputStream.is_open()){
                while(getline (inputStream, currLine)){
                    for(int i = 0; i < Dimensions.cols; ++i){
                        currState = (int) currLine.at(i) - 48;
                        residents[currRow][i].setState(currState);
                    }
                    currRow ++;
                }
                cout << endl;
                inputStream.close();
            }
        }

        ~Neighborhood(){
            for(int i = 0; i < Dimensions.rows; ++i){
                delete[] residents[i];
            }
            delete[] residents;
        }

        void setResidentState(int x, int y){
            residents[x][y].setState(1);
        }

        int getResidentState(int x, int y){
            return residents[x][y].getState();
        }

        void copy(Neighborhood *original){
            for(int r = 0; r < Dimensions.rows; ++r){
                for(int c = 0; c < Dimensions.cols; ++c){
                    residents[r][c].setState(original->getResidentState(r,c)); 
                }
            }
        }

        void evolve(Neighborhood *previous){
            // for(int r = 0; r < Dimensions.rows; ++r){
            //     for(int c = 0; c < Dimensions.cols; ++c){
            //         if(
            //         cout << residents[r][c]; 
            //     }
            //     cout << endl;
            // }
        }

        void updateResident(){

        }

        void print(){
            if(residents){
                for(int r = 0; r < Dimensions.rows; ++r){
                    for(int c = 0; c < Dimensions.cols; ++c){
                        cout << residents[r][c]; 
                    }
                    cout << endl;
                }
            }
            cout << endl;
        }
};

class Validator{

};

class Executor{
    public:
        Executor(){}

        void execute();
};

void checkDimensions(const string inputFile){
    string currLine;
    ifstream inputStream (inputFile);
    if(inputStream.is_open()){
        while(getline (inputStream, currLine)){
            Dimensions.rows++;
        }
        inputStream.close();
    }
    Dimensions.cols = currLine.length();
}

struct InitData {
    string inputFile;
    string outputFile;
    int steps;
    int threads;
} InitData;

int main(int argc, char **argv){
    //read and validate input-------------------
    //validate();
    //init();
    //execute();

    if(argc != 5){
        cout << "program requires four separate args: \n" << "(1) inpute file name, (2) output filename, (3) number of steps, (4) number of threads" << "\n";
        return 0;
    }

    InitData.inputFile = argv[1];
    InitData.outputFile = argv[2];
    InitData.steps = atoi(argv[3]);
    InitData.threads = atoi(argv[4]);
    checkDimensions(InitData.inputFile);
    cout << Dimensions.rows << Dimensions.cols << endl;
    //------------------------------------------

    //create reference board from initial input
    //iterate based on a previous board
    //copy iteration to reference and repeat OR end
    Neighborhood *referenceNeighborhood = new Neighborhood(InitData.inputFile);
    referenceNeighborhood->print();
    
    cout << endl;
    
    Neighborhood *workingNeighborhood = new Neighborhood();
    workingNeighborhood->print();
    workingNeighborhood->copy(referenceNeighborhood);
    workingNeighborhood->print();
    //read file into a neighborhood;
    delete referenceNeighborhood;
    delete workingNeighborhood;


    return 0;
}