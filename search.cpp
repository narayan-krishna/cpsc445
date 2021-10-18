#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>

using namespace std;

/* a simple struct to hold user data for searching*/
struct SearchData{
    string keyword_list;
    string text;
    string output_file_name;
};

/*a search class to declare and run a search based on given data*/
class Search{
public:

/*holds data, a map of keywords to their respective occurence count,
  and a vector of lines of the file to search*/
SearchData data;
map<string, int> keywords;
vector<string> text_lines;

/*construct a search holding the data, and estabblishing the map
  and vector contents based on file processing functions*/
Search(SearchData d) {
    data = d;
    process_keyword_file();
    process_text_file();
}

/*search certain lines (defined in a vector) and add the word counts
  to an input map*/
void search_lines_threads(vector<int> &line_nums, map<string, int> &counters) {
    for(int n : line_nums) { //for every line specified

        string current_line = text_lines[n]; //establish current line in file
        int current_line_len = current_line.length(); //get length

        int i = 0; int j = 0; //establish two pointers
        while(i < current_line_len && j < current_line_len) {
            /*if j is the end of a word or file: */
            if(current_line.at(j) == ' ' || current_line.at(j) == '.') {
                /*establish a word from index of i to j*/
                string word = current_line.substr(i, j - i);
                /*if its established in the map, increment counter*/
                if(keywords.find(word) != keywords.end()) {
                    counters[word]++;
                }
                /*move on. set i (the "beginning" of the word) to the 
                  end of the previous word*/
                j++;
                i = j;
            }
            /*increment j forward*/
            j ++;
        }
    }
}

/*given a map holding results, add them to the map held by search*/
void combine_results(map<string, int> &counters) {
    for(auto k : counters) {
        keywords[k.first] += k.second;
    }
}

/*initialize a map copy that threads can use*/
void init_thread_map(map<string, int> &counters) {
    for(auto k : keywords) {
        counters[k.first] = 0;
    }
}

/*get the size of the text (by line count)*/
int get_text_size() {
    return text_lines.size();
}

/*add keywords with 0 counts to the map*/
void process_keyword_file() {
    string curr_line; 

    ifstream input_stream (data.keyword_list);
    if(input_stream.is_open()) {

        while(getline (input_stream, curr_line)) {
            keywords.insert({curr_line, 0});
        }
        input_stream.close();
    }
}

/*add file lines to vector*/
void process_text_file() {
    string curr_line;
    bool end_char;

    ifstream input_stream (data.text);
    if(input_stream.is_open()) {

        while(getline (input_stream, curr_line)) {
            int end = curr_line.length() - 1;
            /*is the ending character white space or a period*/
            end_char = curr_line.at(end) == ' ' || curr_line.at(end) == '.';
            /*if neither, add an end character (blank) to simplify search*/
            text_lines.push_back(curr_line + (end_char ? "" : " "));             
        }
        input_stream.close();
    }
}

/*some print functionality*/

void print_text() {
    for (auto &n : text_lines) {
        cout << n << endl;
    }
}

void print_keywords() {
    for (auto &n : keywords) {
        cout << n.first << ", " << n.second << endl;
    }
}

void print_keywords_to_file() {
    ofstream outfile;
    outfile.open(data.output_file_name, fstream::app);
    for (auto &n : keywords) {
        outfile << n.first << " " << n.second << endl;
    }
    outfile.close();
}


/*nothing to destroy, no heap usage in map*/
~Search(){}

};


/*executor class to execute search over multiple threads*/
class Executor{
private:
    vector<thread*> threads;
    int thread_count;

    void thread_task(int rank, Search &s) {
        /*a mutex for locking critical section (gathering)*/
        mutex m;
        vector<int> alloc_lines;
        map<string, int> thread_map;
        s.init_thread_map(thread_map);

        /*calculate respective lines based on rank, add to alloc_lines vec*/
        for (int line_num = 0; line_num < s.get_text_size(); line_num ++) {
            if(line_num % thread_count == rank) {
                alloc_lines.push_back(line_num);
            }
        }
        /*because lines are distributed among threads, don't lock search*/
        s.search_lines_threads(alloc_lines, thread_map); //run search
        {
            lock_guard<mutex> lock(m);
            /*gather results*/
            s.combine_results(thread_map);
        }
    }

public:
    /*construct an executor with given thread count*/
    Executor(int thread_count){
       this->thread_count = thread_count; 
    }

    ~Executor(){}

    /*run over specified number of threads*/
    void execute(Search &s) {
        /*instantiate threads*/
        for(int i = 0; i < thread_count; i++) {
            threads.push_back(new thread([&, i]{ thread_task(i, s); }));
        }

        /*delete threads as they finish*/
        for(int i = 0; i < thread_count; i++) {
            thread& t = *threads[i];
            t.join();
            delete threads[i];
        }
        
        /*execution finished*/
    }
};



int main(int argc, char **argv) {
    /*double check input*/
    if (argc != 5) {
        cout << "incorrect input count" << endl;
        return 0;
    }

    /*store inputs as either search data...*/
    SearchData search_input = { argv[1], argv[2], argv[3] };
    /*...or thread count data*/
    int thread_count = stoi(argv[4]);

    Search s(search_input);
    Executor e(thread_count);

    /*create a search s -> execute over multiple threads with exec e*/
    e.execute(s);
    s.print_keywords_to_file();

    return 0;
}
