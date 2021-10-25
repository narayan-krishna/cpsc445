mpiCC dna_genes.cpp
time mpirun -n 10 ./a.out
cat output.txt
rm output.txt