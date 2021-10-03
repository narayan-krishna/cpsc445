g++ search.cpp -lpthread
cat > keywords.txt <<EOF
dog
mouse
hippo
EOF
cat > sometext.txt <<EOF
The mouse was a friend of the dog.
The mouse was in the zoo with the hippo.
The hippo was a very big hippo.
EOF
./a.out keywords.txt sometext.txt output.txt 2
# cat output.txt
# dog 1
# hippo 3
# mouse 2

rm keywords.txt sometext.txt