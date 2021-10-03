g++ life.cpp -lpthread -o life.exe
cat > input.txt <<EOF
0000000001
0000010001
0000101001
0000000000
0000000000
EOF
./life.exe input.txt output.txt 19 30000
cat output.txt
rm output.txt
