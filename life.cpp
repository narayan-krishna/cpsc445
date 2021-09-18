#include <iostream>
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

        void get_state(){
            cout << state << endl;
        }
};

class Neighborhood{
    private:
        Resident *population = nullptr;

    public:
        Neighborhood(int *grid){
            
        }
        ~Neighborhood(){}

        void copy(){}
};

int main(int argc, char **argv){
    Resident r = Resident(ALIVE);
    r.get_state();
    cout << "hello world" << endl;
    return 0;
}