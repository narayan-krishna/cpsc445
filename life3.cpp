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
            state = DEAD;
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
                inputStream.close();
            }
        }

        ~Neighborhood(){
            for(int i = 0; i < Dimensions.rows; ++i){
                delete[] residents[i];
            }
            delete[] residents;
            cout << "Neighborhood: destroyed" << endl;
        }

        void killResident(const int &x, const int &y){
            residents[x][y].kill();
        }

        void animateResident(const int &x, const int &y){
            residents[x][y].animate();
        }

        int getResidentState(const int &x, const int &y){
            return residents[x][y].getState();
        }

        void getResidentCoords(const int &index, int &xCoord, int &yCoord){
            xCoord = (index % Dimensions.rows);
            yCoord = (index / Dimensions.rows); 
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
        }
    
        void printToFile(const string &outputFile){
            ofstream outfile;
            outfile.open (InitData.outputFile, fstream::app);
            if(residents){
                for(int r = 0; r < Dimensions.rows; ++r){
                    for(int c = 0; c < Dimensions.cols; ++c){
                        outfile << residents[r][c]; 
                    }
                    outfile << endl;
                }
            }
            outfile << endl;
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

//an object built for performing simulations using neighborhoods
class Simulation{
    private:
        Neighborhood *referenceNeighborhood;
        Neighborhood *workingNeighborhood;

        int countNeighbors(const int &x, const int &y){
            // bool rightBorder = (y == rows - 1);
            // bool leftBorder = (y == 0);
            // bool topBorder = (x == 0);
            // bool botBorder = (x == cols - 1);
            // for(int i = 0; i < x-1; ++
            return 0;
        }

        void evolveResident(const int &x, const int &y){
            // int neighborCount = countNeighbors(x, y);
            // if(neighborCount < 2 || neighborCount > 3){
            //     workingNeighborhood->killResident(x, y);
            // }else if(neighborCount == 3){
            //     workingNeighborhood->animateResident(x, y);
            // }
            // workingNeighborhood->killResident(x, y);
            workingNeighborhood->animateResident(x, y);
        }

    public:
        Simulation(string inputFile){
            referenceNeighborhood = new Neighborhood(InitData.inputFile);
            workingNeighborhood = new Neighborhood(InitData.inputFile);
        }

        ~Simulation(){
            delete referenceNeighborhood;
            delete workingNeighborhood;
            cout << "Simulation: destroyed" << endl;
        }

        //evolve a neighborhood by the rules of the game of life
        void evolve(){
            for(int r = 0; r < Dimensions.rows; ++r){
                for(int c = 0; c < Dimensions.cols; ++c){
                    evolveResident(r, c);
                }
            }
        }

        void evolveRange(int index_start, int index_end){
            int xCoord, yCoord;
            for(int r = index_start; r < index_end; ++r){
                workingNeighborhood->getResidentCoords(r, xCoord, yCoord);
                evolveResident(xCoord, yCoord);
            }
        }

        void print(){
            int x = 0; int y = 0;
            workingNeighborhood->print();
            cout << endl;
        }
};

class Executor{
    public:
        Simulation *s;

};

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
    //read file into a neighborhood;

    Simulation s = Simulation(InitData.inputFile);
    s.evolveRange(0, 10);
    s.print();


    return 0;
}