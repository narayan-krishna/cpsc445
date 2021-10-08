#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>

using namespace std;


struct SearchData{
    string keyword_list;
    string text;
    string output_file_name;
};

class Search{
public:

SearchData data;
map<string, int> keywords;
vector<string> text_lines;


Search(SearchData d) {
    data = d;
    process_keyword_file();
    process_text_file();
}

void search_lines_threads(vector<int> &line_nums, map<string, int> &counters) {
    for(int n : line_nums) {

        string current_line = text_lines[n];
        int current_line_len = current_line.length();

        // for(int j = 0; j < current_line_len; j++) {
        int i = 0; int j = 0;
        while(i < current_line_len && j < current_line_len) {
            if(current_line.at(j) == ' ' || current_line.at(j) == '.') {
                string word = current_line.substr(i, j - i);
                if(keywords.find(word) != keywords.end()) {
                    counters[word]++;
                }
                j++;
                i = j;
            }
            j ++;
        }
    }
}

void combine_results(map<string, int> &counters) {
    for(auto k : counters) {
        keywords[k.first] += k.second;
    }
}

void init_thread_map(map<string, int> &counters) {
    for(auto k : keywords) {
        counters[k.first] = 0;
    }
}

int get_text_size() {
    return text_lines.size();
}

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

void process_text_file() {
    string curr_line;
    bool end_char;

    ifstream input_stream (data.text);
    if(input_stream.is_open()) {

        while(getline (input_stream, curr_line)) {
            int end = curr_line.length() - 1;
            end_char = curr_line.at(end) == ' ' || curr_line.at(end) == '.';
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

        for (int line_num = 0; line_num < s.get_text_size(); line_num ++) {
            if(line_num % thread_count == rank) {
                alloc_lines.push_back(line_num);
            }
        }
        s.search_lines_threads(alloc_lines, thread_map);
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
        for(int i = 0; i < thread_count; i++) {
            threads.push_back(new thread([&, i]{ thread_task(i, s); }));
        }

        // cout << "processing..." << endl;

        for(int i = 0; i < thread_count; i++) {
            thread& t = *threads[i];
            t.join();
            delete threads[i];
        }
    }
};



int main(int argc, char **argv) {

    /*double check input*/
    if (argc != 5) {
        cout << "incorrect input count" << endl;
        return 0;
    }

    //cout << "hello from doom emacs" << endl;

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
