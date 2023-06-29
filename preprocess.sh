#!/bin/bash

PREFIX=$1
TRAIN_SIZE=$2
VALID_SIZE=$3
DATA_PATH=${4:-$PWD}

current_dir=$(pwd)

INPUT_PATH="$DATA_PATH/$PREFIX"
OUTPUT_PREFIX=$(echo "$PREFIX" | sed "s/.tsv//g")
OUTPUT_PATH="$DATA_PATH/pretraining/$OUTPUT_PREFIX"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

mkdir -p "$OUTPUT_PATH"

if [[ -d $INPUT_PATH ]]; then
    echo "$INPUT_PATH is a directory, processing splits"
    for SPLIT in train dev test; do
        cp "$INPUT_PATH/$SPLIT.tsv" "$OUTPUT_PATH/"
    done

elif [[ -f $INPUT_PATH ]]; then
    echo "$INPUT_PATH is a file, creating splits"

    split -l $TRAIN_SIZE "$INPUT_PATH" "$INPUT_PATH.train."  
    mv "$INPUT_PATH.train.aa" "$OUTPUT_PATH/train.tsv"

    split -l $VALID_SIZE "$INPUT_PATH.train.ab" "$PREFIX.train.ab."
    rm "$INPUT_PATH.train.ab"
    mv "$INPUT_PATH.train.ab.aa" "$OUTPUT_PATH/dev.tsv"
    mv "$INPUT_PATH.train.ab.ab" "$OUTPUT_PATH/test.tsv"

else
    echo "$INPUT_PATH is not valid"
    exit 1
fi

for SPLIT in train dev test; do
    echo "Extracting inputs for $SPLIT..."
    cut $OUTPUT_PATH/$SPLIT.tsv -f1 > $OUTPUT_PATH/$SPLIT.input0
    cut $OUTPUT_PATH/$SPLIT.tsv -f3 > $OUTPUT_PATH/$SPLIT.input1
    # line below generates $SPLIT.output0, $SPLIT.output1 and $SPLIT.output2
    echo "Extracting labels for $SPLIT..." 
    python $SCRIPT_DIR/extract_edit_labels.py --data $OUTPUT_PATH/$SPLIT.tsv
done

# cd $HOME/storage4/early/early/code/fairseq
for INP in input0 input1 label2; do 
    for SPLIT in train dev; do
        python -m examples.roberta.multiprocessing_bpe_encoder \
            --encoder-json "$DATA_PATH/gpt2_bpe/encoder.json" \
            --vocab-bpe "$DATA_PATH/gpt2_bpe/vocab.bpe"  \
            --inputs "$OUTPUT_PATH/$SPLIT.$INP" \
            --outputs "$OUTPUT_PATH/$SPLIT.$INP.bpe" \
            --workers 60 \
            --keep-empty
    done
done

for INP in input0 input1 label2; do 
    fairseq-preprocess \
        --only-source \
        --trainpref "$OUTPUT_PATH/train.$INP.bpe" \
        --validpref "$OUTPUT_PATH/dev.$INP.bpe" \
        --destdir "$OUTPUT_PATH/data-bin/$INP" \
        --workers 60 \
        --srcdict "$DATA_PATH/gpt2_bpe/dict.txt"
done

# levenshtein operation labels for x_minus
fairseq-preprocess \
    --only-source \
    --trainpref "$OUTPUT_PATH/train.label0" \
    --validpref "$OUTPUT_PATH/dev.label0" \
    --destdir "$OUTPUT_PATH/data-bin/label0" \
    --srcdict "$SCRIPT_DIR/label_dict.txt" \
    --workers 60

# levenshtein operation labels for x_plus
fairseq-preprocess \
    --only-source \
    --trainpref "$OUTPUT_PATH/train.label1" \
    --validpref "$OUTPUT_PATH/dev.label1" \
    --destdir "$OUTPUT_PATH/data-bin/label1" \
    --srcdict "$SCRIPT_DIR/label_dict.txt" \
    --workers 60
