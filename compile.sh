g++ life.cpp -lpthread -o life.exe
cat > input.txt <<EOF
100
100
001
EOF
./life.exe input.txt output.txt 10 1
cat output.txt
rm output.txt
