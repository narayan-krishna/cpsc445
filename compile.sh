g++ life.cpp -lpthread -o life.exe
time ./life.exe input.txt output.txt 10 5 
echo
cat output.txt
rm life.exe output.txt