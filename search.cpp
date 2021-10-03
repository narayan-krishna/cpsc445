#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>

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
    cout << "here" << endl;
    process_keyword_file();
    print_keywords();
    process_text_file();
    print_text();
}

// void search_lines(vector<int> &line_nums) { 
//     for(int n : line_nums) {

//         string current_line = text_lines[n];
//         int current_line_len = current_line.length();

//         int i = 0;
//         for(int j = 0; j < current_line_len; j++) {
//             if(curr_line.at(j) == ' ') {
//                 string word = current_line.substring(i, j);
//                 if(key_words.find(word)) {
//                     key_words[word]++;
//                 }
//                 cout << "word" << endl;
//                 i ++;
//             }
//         }
//     }
// }

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

    ifstream input_stream (data.text);
    if(input_stream.is_open()) {

        while(getline (input_stream, curr_line)) {
            text_lines.push_back(curr_line);             
        }
        input_stream.close();
    }
}

void print_keywords() {
    for (auto &n : keywords) {
        cout << n.first << ", " << n.second << endl;
    }
}

void print_text() {
    for (auto &n : text_lines) {
        cout << n << endl;
    }
}

~Search(){}

};



int main(int argc, char **argv) {

    if (argc != 5) {
        cout << "incorrect input count" << endl;
        return 0;
    }

    SearchData search_input = { argv[1], argv[2], argv[3] };
    int thread_count = stoi(argv[4]);

    Search s(search_input);


    return 0;
}


//planning

//use an unordered map to process the first file
//tbd
//use a vector of vectors to store file -- or maybe I could do it on the fly?