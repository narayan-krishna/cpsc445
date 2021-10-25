mpiCC dna_parse.cpp
time mpirun -n 4 ./a.out
cat output.txt
rm output.txt