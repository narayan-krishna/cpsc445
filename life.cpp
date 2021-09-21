#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace std;

enum cellState { DEAD, ALIVE };

/*Container for grid dimensions*/
struct Dimensions {
    int rows = 0;
    int cols = 0;
} Dimensions;

/*Container for user input data*/
struct InitData {
    string inputFile;
    string outputFile;
    int steps;
    int threads;
} InitData;

/*Resident class with an associated state.*/
/*Resident can be dead or alive*/
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

        /*Manually set the state of resident*/
        void setState(int state) {
            this->state = state;
        }

        int getState() {
            return state;
        }

        /*Overload cout operator for printing purposes*/
        friend ostream& operator<< (ostream& os, const Resident& resident) {
            os << resident.state;
            return os;
        }
};

/*Neighborhood class contains dynamically allocated array of 2d residents*/
/*For neighbor counting purposes, pre-existing grids are built on to 
into a larger grid with bordering dead cells*/
class Neighborhood {

    private:
        /**declare 2d array and the dimensions of the grid
         * of dead cells*/
        Resident **residents;
        int deadrows = Dimensions.rows + 2;
        int deadcols = Dimensions.cols + 2;

    public:
        /*initialize grid*/
        Neighborhood() {
            residents = new Resident *[deadrows];
            for(int i = 0; i < deadrows; ++i) {
                residents[i] = new Resident[deadcols];
            }
        }

        /*copy a file to grid. file is copied inside the dead cell border*/
        void fileToNeighborhood(const string &inputFile) {
            string currLine;
            int currRow = 0;
            int currState = 0;

            /*open an input file stream from user file*/
            ifstream inputStream (inputFile);
            if(inputStream.is_open()) {
                while(getline (inputStream, currLine)) {
                    for(int i = 0; i < Dimensions.cols; ++i) {
                        /*convert char to proper int*/
                        currState = (int) currLine.at(i) - 48;

                        /*place inside border by a +1 offset*/
                        if(currState == 0 || currState == 1) {
                            residents[currRow + 1][i + 1].setState(currState);
                        }
                    }
                    currRow ++;
                }
                inputStream.close();
            }
        }

        /*delete neighborhood and associated 2d array in dynamic memory*/
        ~Neighborhood() {
            for(int i = 0; i < deadrows; ++i) {
                delete[] residents[i];
            }
            delete[] residents;
            cout << "Neighborhood: destroyed" << endl;
        }

        /*manually kill resident in neighborhood*/
        void killResident(const int &x, const int &y) {
            residents[x+1][y+1].kill();
        }

        /*manually animate*/
        void animateResident(const int &x, const int &y) {
            residents[x+1][y+1].animate();
        }

        /*get whether resident is dead or alive*/
        int getResidentState(const int &x, const int &y) {
            return residents[x+1][y+1].getState();
        }

        /*get a residents coordinates based of its index*/
        void getResidentCoords(const int &index, int &xCoord, int &yCoord) {
            xCoord = (index % Dimensions.rows);
            yCoord = (index / Dimensions.rows); 
        }

        /*print to console*/
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
    
        /*print to file*/
        void printToFile() {
            ofstream outfile;
            outfile.open (InitData.outputFile, fstream::app);
            if(residents) {

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
        /*count neighbors of surrounding cell*/
        int countNeighbors(const int &x, const int &y) {
            int neighborCount = 0;
            neighborCount += referenceNeighborhood.getResidentState(x+1, y);
            neighborCount += referenceNeighborhood.getResidentState(x-1, y);
            neighborCount += referenceNeighborhood.getResidentState(x+1, y+1);
            neighborCount += referenceNeighborhood.getResidentState(x-1, y-1);
            neighborCount += referenceNeighborhood.getResidentState(x, y+1);
            neighborCount += referenceNeighborhood.getResidentState(x, y-1);
            neighborCount += referenceNeighborhood.getResidentState(x+1, y-1);
            neighborCount += referenceNeighborhood.getResidentState(x-1, y+1);
            return neighborCount;
        }

        /*evolve resident based of defined simulation rules*/
        void evolveResident(int x, int y) {
            int neighborCount = countNeighbors(x, y);

            /*if a resident has few neighbors or too many, kill*/
            if(neighborCount < 2 || neighborCount > 3) {
                workingNeighborhood.killResident(x,y);
            /*if a dead cell has three neighbors, bring to life*/
            }else if(neighborCount == 3) {
                workingNeighborhood.animateResident(x,y);
            }

        }

    public:
        /*two neighborhoods declared. one is the current being worked on,
        making decisions based on previous iteration of simulation (reference)*/
        Neighborhood referenceNeighborhood;
        Neighborhood workingNeighborhood;

        /*initialize neighborhoods from file*/
        Simulation(const string &inputFile) {

            referenceNeighborhood.fileToNeighborhood(InitData.inputFile);
            workingNeighborhood.fileToNeighborhood(InitData.inputFile);
        }

        /*ensure destruction when simulation goes out of scope*/
        ~Simulation() {
            cout << "Simulation: destroyed" << endl;
        }

        /*evolve entire working neighborhood by previously defined ruleset*/
        void evolve() {
            for(int r = 0; r < Dimensions.rows; ++r) {
                for(int c = 0; c < Dimensions.cols; ++c) {
                    evolveResident(r, c);
                }
            }
        }            

        /*evolve a range of residents given ruleset. useful for partitioning*/
        void evolveRange(int index_start, int index_end) {
            int xCoord, yCoord;
            /*calculate coords based of an index for easier partition*/
            for(int r = index_start; r < index_end; ++r) {
                workingNeighborhood.getResidentCoords(r, xCoord, yCoord);
                evolveResident(xCoord, yCoord);
            }
        }

        /*store the current evolution to the reference for next evo*/
        void storeCurrentState() {
            int currentResidentState;
            for(int r = 0; r < Dimensions.rows; ++r) {
                for(int c = 0; c < Dimensions.cols; ++c) {
                    /*copy residents by state*/                    
                    currentResidentState = workingNeighborhood.getResidentState(r,c);
                    if(currentResidentState == DEAD) {
                        referenceNeighborhood.killResident(r,c);
                    } else {
                        referenceNeighborhood.animateResident(r,c);
                    }

                }
            }
        }

        /*print to sim state*/
        void print() {
            workingNeighborhood.print();
            cout << endl;
        }

        /*print sim state to file*/
        void printToFile() {
            workingNeighborhood.printToFile();
        }
};

/*a class built to execute a simulation by using multiple threads*/
class Executor {
    private:
        /*a vector of threads*/
        /*a vector of corresponding nums representing whether theyve
        finished evolution*/
        vector<thread*> threads;
        vector<size_t>threadEvolutionChecker;

        /*a task for threads to execute*/
        /*a thread's rank is used to determine that range of values
        on which it will operate*/
        void evolveTask(size_t rank, Simulation &s, size_t rows, 
                        size_t cols, size_t threads, size_t steps) {
            
            /*calculate partition by rank, thread count, and grid size*/
            size_t gridSize = rows * cols;
            size_t taskSize = (gridSize/threads) + 
                              ((rank < gridSize%threads)?1:0);
            size_t indexStart = rank*(gridSize/threads) + 
                                min(rank, gridSize%threads);
            size_t indexEnd = indexStart + taskSize;
            
            /*operate evolution over the determined range for the input steps*/
            for(size_t i = 0; i < steps; ++i) {
                /*block if range has already been evolved*/
                /*unlocks when all threads have evolved their ranges*/
                while(threadEvolutionChecker[rank] == 1) {}

                /*range needs to be evolved. proceed here*/
                s.evolveRange(indexStart, indexEnd);
                /*establish that this rank has finished evolution*/
                threadEvolutionChecker[rank] = 1;
            }
        }

    public:
        Executor() {
            /*vector initialized such that all threads operated*/
            threadEvolutionChecker = vector<size_t>(InitData.threads, 0);
        }
            
        ~Executor() {
            cout << "Executor: destroyed" << endl;
        }

        /*execute the simulation to user output. employ threads*/
        void execute(Simulation &s) {
            /*print the initial grid for reference*/
            s.printToFile();

            /*create threads and push them to thread vector*/
            for(size_t i = 0; i < InitData.threads; ++i) {
                threads.push_back(new thread([&,i]() {
                    evolveTask(i, s, Dimensions.rows, Dimensions.cols,
                               InitData.threads, InitData.steps);
                    }));
            }

            /*concurrently, use main thread as a checker*/
            for(size_t j = 0; j < InitData.steps; ++j) {

                /*the checker reaches the thread count when all threads finish*/
                size_t checker = 0;
                while(checker != InitData.threads) {
                    /*check the evolution checker all tasks completing*/
                    for(auto i : threadEvolutionChecker) {
                        checker = checker + i;
                    }

                    /*if they haven't, reset for another check*/
                    if(checker != InitData.threads) {
                        checker = 0;
                    }

                    /*if they have, while loop will be passed*/
                }

                /*print results and begin again*/ 
                s.printToFile();
                s.storeCurrentState();

                /*reset checker vector*/
                fill(threadEvolutionChecker.begin(), 
                    threadEvolutionChecker.end(), 0);
            }

            /*delete threads*/
            for(size_t k = 0; k < InitData.threads; ++k) {
                thread& t = *threads[k];
                t.join();
                delete threads[k]; 
            }

            threads.resize(0);
        }

};

/*a function to check the dimensions of the input grid*/
void checkDimensions(const string &inputFile) {
    string currLine;
    ifstream inputStream (inputFile);
    if(inputStream.is_open()) {

        while(getline (inputStream, currLine)) {
            Dimensions.rows++;

            /*perform some serious voodoo to avoid whitespace*/
            if(Dimensions.cols == 0) {
                char currChar;
                for(int i = 0; i < currLine.length(); ++i) {
                    /*ensure chars are ints*/
                    currChar = currLine.at(i);
                    if(currChar == '0' || currChar == '1') {
                        Dimensions.cols++;
                    }
                }
            }
        }
        inputStream.close();
    }
}

int main(int argc, char **argv) {
    if(argc != 5) {
        cout << "program requires four separate args: \n" << "(1) inpute file name, (2) output filename, (3) number of steps, (4) number of threads" << "\n";
        return 0;
    }

    /*storing command line args*/
    InitData.inputFile = argv[1];
    InitData.outputFile = argv[2];
    InitData.steps = atoi(argv[3]);
    InitData.threads = atoi(argv[4]);
    checkDimensions(InitData.inputFile);
    //------------------------------------------

    cout << "threads: " << InitData.threads << endl;
    cout << "steps: " << InitData.steps << endl;
    cout << "rows: " << Dimensions.rows << endl;
    cout << "cols: " << Dimensions.cols << endl;

    Simulation s = Simulation(InitData.inputFile);
    Executor e = Executor();
    e.execute(s);

    cout << "\nc++ version: " << __cplusplus << "\n" << endl;

    return 0;
}