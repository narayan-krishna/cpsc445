#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>

using namespace std;

/*a struct to hold search data*/
struct SearchData{
    string keyword_list;
    string text;
    string output_file_name;
};

/*an instance of a search:
 *take in a file, and a list of words to search for.*/
class Search{
public:

/*declare storage for data, a map for keyword counts,
 *and a vector text_lines */
SearchData data;
map<string, int> keywords;
vector<string> text_lines;


/*construct a search based on input data*/
Search(SearchData d) {
    data = d;
    process_keyword_file();
    process_text_file();
}

/*search the given lines for keywords. update their counts*/
void search_lines(vector<int> &line_nums) {
    /*use a mutex to protect keywords map when threads try
     *to increment in parallel */
    mutex keyword_protection;

    /*for the specific line nums*/
    for(int n : line_nums) {

        string current_line = text_lines[n]; //grab the line
        int current_line_len = current_line.length(); //grab its length

        int i = 0; int j = 0; //establish two points for indexing
        while(i < current_line_len && j < current_line_len) {
            /*if j is a space (end of word (or line)) or period (end of line)*/
            if(current_line.at(j) == ' ' || current_line.at(j) == '.') {
                /*grab the current subtr */
                string word = current_line.substr(i, j - i);
                if(keywords.find(word) != keywords.end()) {
                    {
                        lock_guard<mutex> lock(keyword_protection);
                        keywords[word]++;
                    }
                }
                /*if its not a word, continue to increment*/
                j++;
                /*set new beginning to current end + 1*/
                i = j;
            }
            /*j always goes to next position*/
            j ++;
        }
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

/*lots of printing functionality!*/

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

class Executor{
private:
    vector<thread*> threads;
    int thread_count;

    void thread_task(int rank, Search &s, int text_size) {
        vector<int> alloc_lines;
        //i is line number, thread_count is thread count
        for (int line_num = 0; line_num < text_size; line_num ++) {
            if(line_num % thread_count == rank) {
                alloc_lines.push_back(line_num);
            }
        }
        s.search_lines(alloc_lines); 
    }

public:
    Executor(int thread_count){
       this->thread_count = thread_count; 
    }

    ~Executor(){}

    void execute(Search &s) {
        int text_size = s.get_text_size();
        for(int i = 0; i < thread_count; i++) {
            threads.push_back(new thread([&, i]{ thread_task(i, s, text_size); }));
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

    if (argc != 5) {
        cout << "incorrect input count" << endl;
        return 0;
    }

    //cout << "hello from doom emacs" << endl;

    SearchData search_input = { argv[1], argv[2], argv[3] };
    int thread_count = stoi(argv[4]);

    Search s(search_input);
    Executor e(thread_count);

    e.execute(s);
    s.print_keywords_to_file();

    return 0;
}
