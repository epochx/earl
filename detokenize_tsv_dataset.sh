#!/bin/bash

INPUT=$1

echo $INPUT

# divide TSV input
for NUM in {1..3};
do
    cut -f"$NUM" "$INPUT" > "$INPUT.$NUM";
done

for NUM in {1..3};
do
    sacremoses -l en -j 4 detokenize < "$INPUT.$NUM" > "$INPUT.$NUM".detok
done

OUTPUT=$(echo "$INPUT" | sed 's/.tsv/.detok.tsv/g')

paste $INPUT.1.detok $INPUT.2.detok $INPUT.3.detok > $OUTPUT

for NUM in {1..3};
do
    rm "$INPUT.$NUM"
    rm "$INPUT.$NUM".detok
done
