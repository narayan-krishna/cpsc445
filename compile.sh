mpiCC dna_genes.cpp
time mpirun -n 5 ./a.out
cat output.txt
rm output.txt