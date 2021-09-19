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

struct InitData {
    string inputFile;
    string outputFile;
    int steps;
    int threads;

    int firstPrint = 0;
} InitData;

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
        Resident **residents;

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

        void killResident(int x, int y){
            residents[x][y]->kill();
        }

        void animateResident(int x, int y){
            residents[x][y]->animate();
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
    
        void printToFile(const string &outputFile){
            ofstream outfile;
            if(InitData.firstPrint = 0){
                outfile.open (InitData.outputFile);
            }else{
                outfile.open (InitData.outputFile, fstream::app);
            }
            if(residents){
                for(int r = 0; r < Dimensions.rows; ++r){
                    for(int c = 0; c < Dimensions.cols; ++c){
                        outfile << residents[r][c]; 
                    }
                    outfile << endl;
                }
            }
            outfile << endl;
            ++InitData.firstPrint;
        }

};

// class Validator{
//     private:
//         char *args;
        
//     public:
//         Validator(char **argv){}
        
//         int checkInputFile(){
            
//         }
// };

class Simulation{
    private:
        Neighborhood *referenceNeighborhood;
        Neighborhood *workingNeighborhood;

        int countNeighbors(const int &x, const int &y){
            //border conditions
        }

        int evolveResident(const int &x, const int &y){
            int neighborCount = countNeighbors(x, y);
            if(neighborCount < 2 || neighborCount > 3){
                workingNeighborhood->killResident(x, y);
            }else if(neighborCount == 3){
                workingNeighborhood->animate(x, y);
            }
        }

    public:
        Simulation(){
            referenceNeighborhood = new Neighborhood(InitData.inputFile);
            workingNeighborhood = new Neighborhood();
        }

        void evolve(){
            for(int r = 0; r < InitData.rows; ++r){
                for(int c = 0; c < InitData.cols; ++c){
                    evolveResident(r, c);
                }
            }
        }
};

class Executor{

void checkDimensions(const string &inputFile){
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

int main(int argc, char **argv){
    //read and validate input-------------------
    //validate();
    //init();
    //execute();
    // Validator *val = new Validator(argv);
    if(argc != 5){
        cout << "program requires four separate args: \n" << "(1) inpute file name, (2) output filename, (3) number of steps, (4) number of threads" << "\n";
        return 0;
    }

    InitData.inputFile = argv[1];
    InitData.outputFile = argv[2];
    InitData.steps = atoi(argv[3]);
    InitData.threads = atoi(argv[4]);
    checkDimensions(InitData.inputFile);
    //------------------------------------------

    //create reference board from initial input
    //iterate based on a previous board
    //copy iteration to reference and repeat OR end
    cout << InitData.firstPrint << endl;
    Neighborhood *referenceNeighborhood = new Neighborhood(InitData.inputFile);
    referenceNeighborhood->printToFile(InitData.outputFile);
    
    Neighborhood *workingNeighborhood = new Neighborhood();
    workingNeighborhood->printToFile(InitData.outputFile);
    cout << InitData.firstPrint << endl;
    //read file into a neighborhood;
    delete referenceNeighborhood;
    delete workingNeighborhood;


    return 0;
}