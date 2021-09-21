g++ life.cpp -lpthread -o life.exe
cat > input.txt <<EOF
0000000000
0000010000
0001100000
0000100000
0000000000
EOF
time ./life.exe input.txt output.txt 8 5
echo
cat output.txt
rm life.exe input.txt output.txt