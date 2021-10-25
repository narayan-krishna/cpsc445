mpiCC dna_genes.cpp
time mpirun -n 3 ./a.out
cat output.txt
rm output.txt