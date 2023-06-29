#! /bin/bash

# DATASETS: x paws wi_plus_locness
# TASKS: default frozen
# MODES: edits roberta edits_x edits_x_no_bow

for DATASET in paws;
do
    for TASK in default;
    do
        for MODE in edits_x_mlm_100;
        do
            echo $DATASET $TASK $MODE;
            bash train_downstream.sh $DATASET $TASK $MODE
        done
    done
done