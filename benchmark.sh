#!/bin/bash

#for i in *.cc ; do sed -i 's/SIZE 1024/SIZE 512/g' $i ; done #XXX to change the matrix size

for i in matmul_* ; do if [[ -x "$i" ]] ; then echo $i; ./$i > $i.out ; fi ; done
grep -r "GFlops" *.out | awk -F '[: ]' '{sub(/^matmul_/, "", $1); sub(/.out/, "", $1); print $1, $4 }' > benchmarks
