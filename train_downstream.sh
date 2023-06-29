#! /bin/bash

DATASET=$1
TASK=$2
MODE=$3

if [ "$MODE" = "edits" ]
then
    CHECKPOINT_PATH="$HOME/data/early/pretraining/insertions.detok/checkpoints/checkpoint_last.pt"    
elif [ "$MODE" = "edits_x" ]
then
    CHECKPOINT_PATH="$HOME/data/early/pretraining/x.detok/checkpoints-100epochs/checkpoint_last.pt"
elif [ "$MODE" = "edits_x_no_bow" ]
then
    CHECKPOINT_PATH="$HOME/data/early/pretraining/x.detok/checkpoints_x_no_bow/checkpoint_last.pt"
elif [ "$MODE" = "edits_x_mlm_100" ]
then
    CHECKPOINT_PATH="$HOME/data/early/pretraining/x.detok/checkpoints_x_mlm_100/checkpoint_last.pt"
else
    CHECKPOINT_PATH="$HOME/data/early/roberta.base/model.pt"
fi

BIN_DATA_PATH="$HOME/data/early/$DATASET-bin"
RESULTS_PATH="$HOME/data/early/results/$DATASET/$TASK/$MODE"

mkdir -p "$RESULTS_PATH"

echo $DATASET
echo $TASK
echo $MODE

echo "fairseq-hydra-train --config-dir configs --config-name "$DATASET"_"$TASK" task.data=$BIN_DATA_PATH checkpoint.restore_file=$CHECKPOINT_PATH checkpoint.save_dir=$RESULTS_PATH | tee -a $RESULTS_PATH/train.log"

fairseq-hydra-train --config-dir configs --config-name "$DATASET"_"$TASK" task.data="$BIN_DATA_PATH" checkpoint.restore_file="$CHECKPOINT_PATH" checkpoint.save_dir="$RESULTS_PATH" | tee -a "$RESULTS_PATH/train.log"

