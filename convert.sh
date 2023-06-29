#!/bin/bash

# wget https://storage.googleapis.com/paws/english/paws_wiki_labeled_final.tar.gz

GLUE_DATA_FOLDER=$1
TASK=$2

echo "Preprocessing $TASK"

if [ "$TASK" = "paws" ]
then
    SPLITS="train dev test"
    INPUT_COUNT=2
    INPUT_COLUMNS=( 2 3 )
    TEST_INPUT_COLUMNS=( 2 3 )
    LABEL_COLUMN=4
elif [ "$TASK" = "x" ]
then
    SPLITS="train dev test"
    INPUT_COUNT=2
    INPUT_COLUMNS=( 1 3 )
    TEST_INPUT_COLUMNS=( 1 3 )
    LABEL_COLUMN=2
elif [ "$TASK" = "x.detok" ]
then
    SPLITS="train dev test"
    INPUT_COUNT=2
    INPUT_COLUMNS=( 1 3 )
    TEST_INPUT_COLUMNS=( 1 3 )
    LABEL_COLUMN=2
elif [ "$TASK" = "wi_plus_locness" ]
then
    SPLITS="train dev test"
    INPUT_COUNT=2
    INPUT_COLUMNS=( 1 3 )
    TEST_INPUT_COLUMNS=( 1 3 )
    LABEL_COLUMN=2
fi

TASK_DATA_FOLDER="$GLUE_DATA_FOLDER/$TASK"


rm -rf "$TASK_DATA_FOLDER/processed"
mkdir -p "$TASK_DATA_FOLDER/processed"
for SPLIT in $SPLITS
do
    if [ "$TASK" = "paws" ]
    then
        # remove headers in case of paws
        tail -n +2 "$TASK_DATA_FOLDER/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
    else
        if [ "$SPLIT" = "dev" ]
        then
            cat "$TASK_DATA_FOLDER/valid.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";

        else
            cat "$TASK_DATA_FOLDER/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
        fi
    fi
    cp "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv";
    rm "$TASK_DATA_FOLDER/processed/$SPLIT.tsv.temp";
done

# Split into input0, input1 and label
for SPLIT in $SPLITS
do
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
        if [[ "$SPLIT" != test* ]]
        then
            COLUMN_NUMBER=${INPUT_COLUMNS[$INPUT_TYPE]}
        else
            COLUMN_NUMBER=${TEST_INPUT_COLUMNS[$INPUT_TYPE]}
        fi
      cut -f"$COLUMN_NUMBER" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.raw.input$INPUT_TYPE";
    done

    if [[ "$SPLIT" != test* ]]
    then
        cut -f"$LABEL_COLUMN" "$TASK_DATA_FOLDER/processed/$SPLIT.tsv" > "$TASK_DATA_FOLDER/processed/$SPLIT.label";
    fi

    # BPE encode.
    for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
    do
      LANG="input$INPUT_TYPE"
      echo "BPE encoding $SPLIT/$LANG"
      python $HOME/early/code/fairseq/examples/roberta/multiprocessing_bpe_encoder.py \
      --encoder-json encoder.json \
      --vocab-bpe vocab.bpe \
      --inputs "$TASK_DATA_FOLDER/processed/$SPLIT.raw.$LANG" \
      --outputs "$TASK_DATA_FOLDER/processed/$SPLIT.$LANG" \
      --workers 60 \
      --keep-empty;
    done
done

# Remove output directory.
rm -rf "$TASK-bin"

DEVPREF="$TASK_DATA_FOLDER/processed/dev.LANG"
TESTPREF="$TASK_DATA_FOLDER/processed/test.LANG"

# Run fairseq preprocessing:
for INPUT_TYPE in $(seq 0 $((INPUT_COUNT-1)))
do
    LANG="input$INPUT_TYPE"
    fairseq-preprocess \
        --only-source \
        --trainpref "$TASK_DATA_FOLDER/processed/train.$LANG" \
        --validpref "${DEVPREF//LANG/$LANG}" \
        --testpref "${TESTPREF//LANG/$LANG}" \
        --destdir "$TASK-bin/$LANG" \
        --workers 60 \
        --srcdict dict.txt;
done

fairseq-preprocess \
    --only-source \
    --trainpref "$TASK_DATA_FOLDER/processed/train.label" \
    --validpref "${DEVPREF//LANG/label}" \
    --destdir "$TASK-bin/label" \
    --workers 60;
