g++ search.cpp -lpthread
cat > keywords.txt <<EOF
cat
dog
mouse
hippo
zebra
lion
penguin
snake
EOF
cat > sometext.txt <<EOF
The lion saw a zebra and a penguin and a mouse
The penguin saw a zebra and a zebra
The zebra saw a lion and a penguin and a zebra
The hippo saw a zebra
The dog saw a hippo and a zebra
The mouse saw a hippo
The hippo saw a penguin
The dog saw a zebra and a lion and a hippo and a lion
EOF
valgrind --tool=memcheck ./a.out keywords.txt sometext.txt output.txt 4
cat output.txt

rm keywords.txt sometext.txt output.txt
