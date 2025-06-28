#!/bin/bash

##for i in *.cc ; do sed -i 's/SIZE 1024/SIZE 512/g' $i ; done #XXX to change the matrix size
#
#for i in matmul_* ; do if [[ -x "$i" ]] ; then echo $i; ./$i > $i.out ; fi ; done
##grep -r "GFlops" *.out | awk -F '[: ]' '{sub(/^matmul_/, "", $1); sub(/.out/, "", $1); print $1, $4 }' > benchmarks
#grep -r "Time" *.out | awk -F '[: ]' '{sub(/^matmul_/, "", $1); sub(/.out/, "", $1); print $1, $5 }' > benchmarks

# Output file
echo -e "Kernel\tSize\tTime(ms)\tGFLOPs" > benchmarks.tsv

# Benchmark each executable
for exe in matmul_*; do
    if [[ -x "$exe" ]]; then
        sleep 2s
        echo "Running $exe"
        ./"$exe" > "$exe.out"

        # Extract SIZE from filename (assuming format: matmul_<name>_size<value>)
        kernel=$(echo "$exe" | sed -E 's/matmul_//; s/_size[0-9]+//')
        size=$(echo "$exe" | grep -oE 'size[0-9]+' | cut -c5-)

        # Extract time and GFLOPs
        time=$(grep "Time" "$exe.out" | awk '{print $NF}')
        gflops=$(grep "Performance" "$exe.out" | awk '{print $(NF -1)}')

        # Default values if parsing fails
        time=${time:-NA}
        gflops=${gflops:-NA}

        echo -e "$kernel\t$size\t$time\t$gflops" >> benchmarks.tsv
    fi
done

# Display sorted by GFLOPs
echo -e "\nSorted by performance (GFLOPs):"
sort -k4 -n -r benchmarks.tsv | column -t

