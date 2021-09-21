#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
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

class Resident {
    private:
        int state;

    public:
        Resident(int stated = DEAD) {
           state = stated;
        }

        void animate() {
            state = ALIVE;
        }

        void kill() {
            state = DEAD;
        }

        void setState(int state) {
            this->state = state;
        }

        int getState() {
            return state;
        }

        friend ostream& operator<< (ostream& os, const Resident& resident) {
            os << resident.state;
            return os;
        }

};

class Neighborhood {
    private:
        Resident **residents;
        int deadrows = Dimensions.rows + 2;
        int deadcols = Dimensions.cols + 2;

    public:
        Neighborhood() {
            residents = new Resident *[deadrows];
            for(int i = 0; i < deadrows; ++i) {
                residents[i] = new Resident[deadcols];
            }
        }

        void fileToNeighborhood(const string &inputFile) {
            string currLine;
            int currRow = 0;
            int currState = 0;

            ifstream inputStream (inputFile);
            if(inputStream.is_open()) {
                while(getline (inputStream, currLine)) {
                    for(int i = 0; i < Dimensions.cols; ++i) {
                        currState = (int) currLine.at(i) - 48;
                        if(currLine.at(i) != '0' && currLine.at(i) != '1'){
                            cout << "X";
                        }
                        if(currState == 0 || currState == 1) {
                            residents[currRow + 1][i + 1].setState(currState);
                        }
                    }
                    currRow ++;
                }
                inputStream.close();
            }
        }

        ~Neighborhood() {
            for(int i = 0; i < deadrows; ++i) {
                delete[] residents[i];
            }
            delete[] residents;
            cout << "Neighborhood: destroyed" << endl;
        }

        void killResident(const int &x, const int &y) {
            residents[x+1][y+1].kill();
        }

        void animateResident(const int &x, const int &y) {
            residents[x+1][y+1].animate();
        }

        int getResidentState(const int &x, const int &y) {
            return residents[x+1][y+1].getState();
        }

        void getResidentCoords(const int &index, int &xCoord, int &yCoord) {
            xCoord = (index % Dimensions.rows);
            yCoord = (index / Dimensions.rows); 
        }

        void print() {
            if(residents) {
                for(int r = 1; r < Dimensions.rows + 1; ++r) {
                    for(int c = 1; c < Dimensions.cols + 1; ++c) {
                        cout << residents[r][c]; 
                    }
                    cout << endl;
                }
            }
        }
    
        void printToFile() {
            ofstream outfile;
            outfile.open (InitData.outputFile, fstream::app);
            if(residents){
                for(int r = 1; r < Dimensions.rows + 1; ++r) {
                    for(int c = 1; c < Dimensions.cols + 1; ++c) {
                        outfile << residents[r][c]; 
                    }
                    outfile << endl;
                }
            }
            outfile << endl;
        }

};

//an object built for performing simulations using neighborhoods
class Simulation {
    private:


    public:
        Neighborhood referenceNeighborhood;
        Neighborhood workingNeighborhood;
        Simulation(const string &inputFile) {

            referenceNeighborhood.fileToNeighborhood(InitData.inputFile);
            workingNeighborhood.fileToNeighborhood(InitData.inputFile);
        }

        ~Simulation() {
            cout << "Simulation: destroyed" << endl;
        }

        //evolve a neighborhood by the rules of the game of life
        void evolve() {
            for(int r = 0; r < Dimensions.rows; ++r) {
                for(int c = 0; c < Dimensions.cols; ++c) {
                    evolveResident(r, c);
                }
            }
        }            

        void evolveRange(int index_start, int index_end) {
            int xCoord, yCoord;
            for(int r = index_start; r < index_end; ++r) {
                workingNeighborhood.getResidentCoords(r, xCoord, yCoord);
                evolveResident(xCoord, yCoord);
            }
        }
        int countNeighbors(int x, int y) {
            int neighborCount = 0;
            //x + 1, y
            neighborCount += referenceNeighborhood.getResidentState(x+1, y);
            //x - 1, yreferenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x-1, y);
            //x + 1, y+1,referenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x+1, y+1);
            //x - 1, y-1referenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x-1, y-1);
            //x, y+1referenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x, y+1);
            //x, y-1referenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x, y-1);
            //x + 1, y-1referenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x+1, y-1);
            //x - 1, y+1referenceNeighborhood
            neighborCount += referenceNeighborhood.getResidentState(x-1, y+1);
            return neighborCount;
        }

        void evolveResident(int x, int y) {
            int neighborCount = countNeighbors(x, y);
            // int state = workingNeighborhood.getResidentState(x,y);
            // if(state == ALIVE) {
            //     if(neighborCount != 2 && neighborCount != 3) {
            //         workingNeighborhood.killResident(x, y);
            //     }
            // }else if(neighborCount == 3){
            //     workingNeighborhood.animateResident(x,y);
            // }
            if(neighborCount < 2 || neighborCount > 3){
                workingNeighborhood.killResident(x,y);
            }else if(neighborCount == 3){
                workingNeighborhood.animateResident(x,y);
            }
            // if(neighborCount > 0){
            //     workingNeighborhood.animateResident(x,y);
            // }
            // if(workingNeighborhood.getResidentState(x,y) == DEAD){
            //     workingNeighborhood.animateResident(x,y);
            // }else{
            //     workingNeighborhood.killResident(x, y);
            // }
            neighborCount = 0;
        }

        void storeCurrentState() {
            int currentResidentState;
            for(int r = 0; r < Dimensions.rows; ++r) {
                for(int c = 0; c < Dimensions.cols; ++c) {
                    currentResidentState = workingNeighborhood.getResidentState(r,c);
                    if(currentResidentState == DEAD) {
                        referenceNeighborhood.killResident(r,c);
                    }else{
                        referenceNeighborhood.animateResident(r,c);
                    }
                }
            }
        }

        void print() {
            workingNeighborhood.print();
            cout << endl;
        }

        void printToFile() {
            workingNeighborhood.printToFile();
        }
};

class Executor {
    private:
        vector<thread*> threads;
        vector<size_t>threadEvolutionChecker;

        void evolveTask(size_t rank, Simulation &s, size_t rows, 
                        size_t cols, size_t threads, size_t steps) {
            // cout << "Rank " << rank << ": armed and ready" << endl;
            size_t gridSize = rows * cols;
            size_t taskSize = (gridSize/threads) + 
                              ((rank < gridSize%threads)?1:0);
            size_t indexStart = rank*(gridSize/threads) + 
                                min(rank, gridSize%threads);
            size_t indexEnd = indexStart + taskSize;
            // cout << indexStart << ", " << indexEnd << endl;
            int checker = 0;
            for(size_t i = 0; i < steps; ++i) {
                //if this evolution has been completed
                //and stage has not been reset, block
                while(threadEvolutionChecker[rank] == 1) {}
                //other wise, evolve the range
                s.evolveRange(indexStart, indexEnd);
                //then, say that this evolution has been completed
                threadEvolutionChecker[rank] = 1;
            }
        }

    public:

        Executor() {
            threadEvolutionChecker = vector<size_t>(InitData.threads, 0);
        }
            
        ~Executor() {
            cout << "Executor: destroyed" << endl;
        }

        void execute(Simulation &s) {
            // s.workingNeighborhood.animateResident(3,4);
            // s.print();
            s.printToFile();
            for(size_t i = 0; i < InitData.threads; ++i) {
                threads.push_back(new thread([&,i]() {
                    evolveTask(i, s, Dimensions.rows, Dimensions.cols,
                               InitData.threads, InitData.steps);
                    }));
            }

            for(size_t j = 0; j < InitData.steps; ++j) {
                size_t checker = 0;
                while(checker != InitData.threads) {
                    for(auto i : threadEvolutionChecker) {
                        checker = checker + i;
                    }
                    if(checker != InitData.threads) {
                        checker = 0;
                    }
                }
                // s.print();
                s.printToFile();
                s.storeCurrentState();
                fill(threadEvolutionChecker.begin(), 
                    threadEvolutionChecker.end(), 0);
            }

            for(size_t k = 0; k < InitData.threads; ++k) {
                thread& t = *threads[k];
                t.join();
                delete threads[k]; 
            }

            threads.resize(0);
        }

};

void checkDimensions(const string &inputFile) {
    string currLine;
    ifstream inputStream (inputFile);
    if(inputStream.is_open()) {
        while(getline (inputStream, currLine)) {
            Dimensions.rows++;
            if(Dimensions.cols == 0){
                Dimensions.cols = currLine.length();
            }
        }
        inputStream.close();
    }
}

int main(int argc, char **argv) {
    //read and validate input-------------------
    //validate();
    //init();
    //execute();
    // Validator *val = new Validator(argv);
    if(argc != 5) {
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
    cout << "threads: " << InitData.threads << endl;
    cout << "steps: " << InitData.steps << endl;
    cout << "rows: " << Dimensions.rows << endl;
    cout << "cols: " << Dimensions.cols << endl;
    // Neighborhood n = Neighborhood();
    // n.fileToNeighborhood(InitData.inputFile);
    // n.print();
    // n.animateResident(2,3);
    // cout << endl;
    // n.print();
    // cout << n.getResidentState(2,3) << endl;
    // cout << n.getResidentState(3,7) << endl;
    // int x = 2; int y = 3; int count = 0;
    // count += n.getResidentState(x+1, y);
    
    // count += n.getResidentState(x-1, y);
    
    // count += n.getResidentState(x+1, y+1);
    
    // count += n.getResidentState(x-1, y-1);
    
    // count += n.getResidentState(x, y+1);
    
    // count += n.getResidentState(x, y-1);
    
    // count += n.getResidentState(x+1, y-1);
    
    // count += n.getResidentState(x-1, y+1);
    // cout << count + 1<< endl;

    Simulation s = Simulation(InitData.inputFile);
    // s.evolve();
    // s.print();
    Executor e = Executor();
    e.execute(s);
    cout << "\nc++ version: " << __cplusplus << "\n" << endl;

    return 0;
}