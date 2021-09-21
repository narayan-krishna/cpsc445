#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace std;

enum cell_state { DEAD, ALIVE };

/*Container for grid dimensions*/
struct Dimensions {
    int rows = 0;
    int cols = 0;
} Dimensions;

/*Container for user input data*/
struct InitData {
    string input_file;
    string output_file;
    int steps;
    int threads;
} InitData;

/*Resident class with an associated state.*/
/*Resident can be dead or alive*/
class Resident {
    private:
        bool state;

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
        void set_state(bool state) {
            this->state = state;
        }

        bool get_state() {
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
        size_t deadrows = Dimensions.rows + 2;
        size_t deadcols = Dimensions.cols + 2;
        const size_t buffer = 1;

    public:
        /*initialize grid*/
        Neighborhood() {
            residents = new Resident *[deadrows];
            for(size_t i = 0; i < deadrows; ++i) {
                residents[i] = new Resident[deadcols];
            }
        }

        /*overload -- copy a file to grid. file is copied inside the dead cell border*/
        Neighborhood(const string &input_file) {

            /*original construction*/
            residents = new Resident *[deadrows];
            for(size_t i = 0; i < deadrows; ++i) {
                residents[i] = new Resident[deadcols];
            }

            string curr_line;
            int curr_row = 0;
            int curr_state = 0;

            /*open an input file stream from user file*/
            ifstream input_stream (input_file);
            if(input_stream.is_open()) {
                while(getline (input_stream, curr_line)) {
                    for(size_t i = 0; i < Dimensions.cols; ++i) {
                        /*convert char to proper int*/
                        curr_state = (int) curr_line.at(i) - 48;

                        /*place inside border by a +1 offset*/
                        if(curr_state == 0 || curr_state == 1) {
                            residents[curr_row + buffer][i + buffer].set_state(curr_state);
                        }
                    }
                    curr_row ++;
                }
                input_stream.close();
            }
        }

        /*copy a file to grid. file is copied inside the dead cell border*/
        void file_to_neighborhood(const string &input_file) {
            string curr_line;
            int curr_row = 0;
            int curr_state = 0;

            /*open an input file stream from user file*/
            ifstream input_stream (input_file);
            if(input_stream.is_open()) {
                while(getline (input_stream, curr_line)) {
                    for(size_t i = 0; i < Dimensions.cols; ++i) {
                        /*convert char to proper int*/
                        curr_state = (int) curr_line.at(i) - 48;

                        /*place inside border by a +1 offset*/
                        if(curr_state == 0 || curr_state == 1) {
                            residents[curr_row + buffer][i + buffer].set_state(curr_state);
                        }
                    }
                    curr_row ++;
                }
                input_stream.close();
            }
        }

        /*delete neighborhood and associated 2d array in dynamic memory*/
        ~Neighborhood() {
            for(size_t i = 0; i < deadrows; ++i) {
                delete[] residents[i];
            }
            delete[] residents;
            cout << "Neighborhood: destroyed" << endl;
        }

        /*manually kill resident in neighborhood*/
        void kill_resident(const int &x, const int &y) {
            residents[x+buffer][y+buffer].kill();
        }

        /*manually animate*/
        void animate_resident(const int &x, const int &y) {
            residents[x+buffer][y+buffer].animate();
        }

        /*get whether resident is dead or alive*/
        int get_resident_state(const int &x, const int &y) {
            return residents[x+buffer][y+buffer].get_state();
        }

        /*get a residents coordinates based of its index*/
        void get_resident_coords(const int &index, int &xCoord, int &yCoord) {
            xCoord = (index % Dimensions.rows);
            yCoord = (index / Dimensions.rows); 
        }

        /*copy from another neighborhood*/
        void copy(Neighborhood &from) {
            int current_resident_state;

            for(size_t r = 0; r < Dimensions.rows; ++r) {
                for(size_t c = 0; c < Dimensions.cols; ++c) {
                    /*copy residents by state*/                    
                    current_resident_state = from.get_resident_state(r,c);
                    if(current_resident_state == DEAD) {
                       kill_resident(r,c);
                    } else {
                       animate_resident(r,c);
                    }

                }
            }
        }

        /*print to console*/
        void print() {
            if(residents) {
                for(size_t r = 1; r < Dimensions.rows + 1; ++r) {
                    for(size_t c = 1; c < Dimensions.cols + 1; ++c) {
                        cout << residents[r][c]; 
                    }
                    cout << endl;
                }
            }
        }
    
        /*print to file*/
        void print_to_file() {

            /*create output stream to outfile, use resident operator overload*/
            ofstream outfile;
            outfile.open (InitData.output_file, fstream::app);
            if(residents) {
                
                for(size_t r = 1; r < Dimensions.rows + 1; ++r) {
                    for(size_t c = 1; c < Dimensions.cols + 1; ++c) {
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
        int count_neighbors(const int &x, const int &y) {
            auto neighbor_count = 0;
            neighbor_count += reference_neighborhood.get_resident_state(x+1, y);
            neighbor_count += reference_neighborhood.get_resident_state(x-1, y);
            neighbor_count += reference_neighborhood.get_resident_state(x+1, y+1);
            neighbor_count += reference_neighborhood.get_resident_state(x-1, y-1);
            neighbor_count += reference_neighborhood.get_resident_state(x, y+1);
            neighbor_count += reference_neighborhood.get_resident_state(x, y-1);
            neighbor_count += reference_neighborhood.get_resident_state(x+1, y-1);
            neighbor_count += reference_neighborhood.get_resident_state(x-1, y+1);
            return neighbor_count;
        }

        /*evolve resident based of defined simulation rules*/
        void evolve_resident(int x, int y) {
            auto neighbor_count = count_neighbors(x, y);

            /*if a resident has few neighbors or too many, kill*/
            if(neighbor_count < 2 || neighbor_count > 3) {
                working_neighborhood.kill_resident(x,y);
            /*if a dead cell has three neighbors, bring to life*/
            }else if(neighbor_count == 3) {
                working_neighborhood.animate_resident(x,y);
            }

        }

    public:
        /*two neighborhoods declared. one is the current being worked on,
        making decisions based on previous iteration of simulation (reference)*/
        Neighborhood reference_neighborhood;
        Neighborhood working_neighborhood;

        /*initialize neighborhoods from file*/
        Simulation(const string &input_file) {
            reference_neighborhood.file_to_neighborhood(InitData.input_file);
            working_neighborhood.file_to_neighborhood(InitData.input_file);
        }

        Simulation(Neighborhood &input_neighborhood) {
            reference_neighborhood.copy(input_neighborhood);
            working_neighborhood.copy(input_neighborhood);
        }

        /*ensure destruction when simulation goes out of scope*/
        ~Simulation() {
            cout << "Simulation: destroyed" << endl;
        }

        /*evolve entire working neighborhood by previously defined ruleset*/
        void evolve() {
            for(int r = 0; r < Dimensions.rows; ++r) {
                for(int c = 0; c < Dimensions.cols; ++c) {
                    evolve_resident(r, c);
                }
            }
        }            

        /*evolve a range of residents given ruleset. useful for partitioning*/
        void evolve_range(int index_start, int index_end) {
            int x_coord, y_coord;
            /*calculate coords based of an index for easier partition*/
            for(size_t r = index_start; r < index_end; ++r) {
                working_neighborhood.get_resident_coords(r, x_coord, y_coord);
                evolve_resident(x_coord, y_coord);
            }
        }

        /*store the current evolution to the reference for next evo*/
        void store_current_state() {
            int current_resident_state;

            for(size_t r = 0; r < Dimensions.rows; ++r) {
                for(size_t c = 0; c < Dimensions.cols; ++c) {
                    /*copy residents by state*/                    
                    current_resident_state = working_neighborhood.get_resident_state(r,c);
                    if(current_resident_state == DEAD) {
                        reference_neighborhood.kill_resident(r,c);
                    } else {
                        reference_neighborhood.animate_resident(r,c);
                    }

                }
            }
        }

        /*print to sim state*/
        void print() {
            working_neighborhood.print();
            cout << endl;
        }

        /*print sim state to file*/
        void print_to_file() {
            working_neighborhood.print_to_file();
        }
};

/*a class built to execute a simulation by using multiple threads*/
class Executor {

    private:
        /*a vector of threads*/
        /*a vector of corresponding nums representing whether theyve
        finished evolution*/
        vector<thread*> threads;
        vector<size_t>thread_evolution_checker;

        /*a task for threads to execute*/
        /*a thread's rank is used to determine that range of values
        on which it will operate*/
        void evolve_task(size_t rank, Simulation &s, size_t rows, 
                        size_t cols, size_t threads, size_t steps) {
            
            /*calculate partition by rank, thread count, and grid size*/
            size_t grid_size = rows * cols;
            size_t task_size = (grid_size/threads) + 
                              ((rank < grid_size%threads)?1:0);
            size_t index_start = rank*(grid_size/threads) + 
                                min(rank, grid_size%threads);
            size_t index_end = index_start + task_size;
            // cout << index_start << ", " << index_end << endl;
            
            /*operate evolution over the determined range for the input steps*/
            for(size_t i = 0; i < steps; ++i) {
                // /*block if range has already been evolved*/
                /*unlocks when all threads have evolved their ranges*/
                while(thread_evolution_checker[rank] == 1) {}

                /*range needs to be evolved. proceed here*/
                s.evolve_range(index_start, index_end);
                /*establish that this rank has finished evolution*/
                thread_evolution_checker[rank] = 1;
            // }
            }
        }

    public:

        Executor() {
            /*vector initialized such that all threads operated*/
            thread_evolution_checker = vector<size_t>(InitData.threads, 0);
        }
            
        ~Executor() {
            cout << "Executor: destroyed" << endl;
        }

        /*execute the simulation to user output. employ threads*/
        void execute(Simulation &s) {
            /*print the initial grid for reference*/
            s.print_to_file();

            /*allocating thread for each step -- much faster*/
            // for(size_t st = 0; st < InitData.steps; ++st){

            //     /*create threads and push them to thread vector*/
            //     for(size_t i = 0; i < InitData.threads; ++i) {
            //         threads.push_back(new thread([&,i]() {
            //             evolve_task(i, s, Dimensions.rows, Dimensions.cols,
            //                     InitData.threads, InitData.steps);
            //             }));
            //     }

            //     s.print_to_file();
            //     s.store_current_state();

            //     for(size_t k = 0; k < InitData.threads; ++k) {
            //         thread& t = *threads[k];
            //         t.join();
            //         delete threads[k]; 
            //         // cout << "Thread: destroyed" << endl;
            //     }

            //     threads.resize(0);
            // }

            /*or locking them using the main program*/
            for(size_t i = 0; i < InitData.threads; ++i) {
                threads.push_back(new thread([&,i]() {
                    evolve_task(i, s, Dimensions.rows, Dimensions.cols,
                               InitData.threads, InitData.steps);
                    }));
            }

            for(size_t j = 0; j < InitData.steps; ++j) {
                /*the checker reaches the thread count when all threads finish*/
                int checker = 0;
                while(checker != InitData.threads) {
                    /*check the evolution checker all tasks completing*/
                    for(auto i : thread_evolution_checker) {
                        checker = checker + i;
                    }
                    /*if they haven't, reset for another check*/
                    if(checker != InitData.threads) {
                        checker = 0;
                    }
                    /*if they have, while loop will be passed*/
                }

                /*print results and begin again*/ 
                s.print_to_file();
                s.store_current_state();

                /*reset checker vector*/
                fill(thread_evolution_checker.begin(), 
                    thread_evolution_checker.end(), 0);
                // checker = 0;
            }

            /*delete threads*/
            for(size_t k = 0; k < InitData.threads; ++k) {
                thread& t = *threads[k];
                t.join();
                delete threads[k]; 
                cout << "Thread: destroyed" << endl;
            }

            threads.resize(0);
        }
                        
};

/*a function to check the dimensions of the input grid*/
void checkDimensions(const string &input_file) {

    string curr_line;
    ifstream input_stream (input_file);
    if(input_stream.is_open()) {

        while(getline (input_stream, curr_line)) {
            Dimensions.rows++;

            /*perform some serious voodoo to avoid whitespace*/
            if(Dimensions.cols == 0) {
                char curr_char;
                for(size_t i = 0; i < curr_line.length(); ++i) {
                    /*ensure chars are ints*/
                    curr_char = curr_line.at(i);
                    if(curr_char == '0' || curr_char == '1') {
                        Dimensions.cols++;
                    }
                }
            }
        }
        input_stream.close();
    }
}

class Validator {
    private:
        /*validate input name using ifstream*/
        bool check_input_name() {
            ifstream test(InitData.input_file);
            if(!test) {
                cout << "input file invalid" << endl;
                return false;
            }
            return true;
        }

        /*check that steps are 0 or more*/
        bool check_steps() {
            if(InitData.steps < 0) {
                cout << "invalid step count" << endl;
                return false;
            }
            return true;
        }

        /*check that steps are 0 or more*/
        bool check_threads() { 
            if(InitData.threads < 0) {
                cout << "invalid thread count" << endl;
                return false;
            }
            return true;
        }

    public:
        Validator() {}

        /*all checker needs to be validate*/
        bool validate() {
            int sumEffort = 0;
            sumEffort += check_input_name();
            sumEffort += check_steps();
            sumEffort += check_threads();

            if(sumEffort == 3) {
                /*make user aware anyway if thread count is greater than total cells*/
                if (InitData.threads > (Dimensions.rows*Dimensions.cols)) {
                    cout << Dimensions.rows*Dimensions.cols << endl;
                    cout << InitData.threads << endl;
                    cout << "ineffecient thread count" << endl;
                }
                return true;
            }
            return false;
        }
};


int main(int argc, char **argv) {

    if(argc != 5) {
        cout << "program requires four separate args: \n" << "(1) inpute file name, (2) output filename, (3) number of steps, (4) number of threads" << "\n";
        return 0;
    }

    /*storing command line args*/
    InitData.input_file = argv[1];
    InitData.output_file = argv[2];
    InitData.steps = atoi(argv[3]);
    InitData.threads = atoi(argv[4]);
    checkDimensions(InitData.input_file);

    /*validating input*/
    Validator v = Validator();
    if(v.validate() == false){
        return 0;
    }
    
    /*printing some states to command line*/
    cout << "threads: " << InitData.threads << endl;
    cout << "steps: " << InitData.steps << endl;
    cout << "rows: " << Dimensions.rows << endl;
    cout << "cols: " << Dimensions.cols << endl << endl;
    
    /*make neighborhood -> enter it into simulator -> execute simulation*/
    auto neighborhood = Neighborhood(InitData.input_file);
    auto simulation = Simulation(neighborhood);
    auto executor = Executor();

    executor.execute(simulation);

    cout << "\nc++ version: " << __cplusplus << "\n" << endl;

    return 0;
}