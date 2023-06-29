#!/bin/bash

PREFIX=$1
DATA_PATH=${2:-$PWD}

INPUT_PATH="$DATA_PATH/pretraining/$PREFIX"
# OUTPUT_PATH="$INPUT_PATH"

OUTPUT_PATH="/scratch/aad13288rp/checkpoints_x_mlm"

TOTAL_NUM_UPDATES=7812  # 10 epochs through IMDB for bsz 32
WARMUP_UPDATES=469      # 6 percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8         # Batch size.
ROBERTA_PATH="$DATA_PATH/roberta.base/model.pt"

fairseq-train "$INPUT_PATH/data-bin" \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 4400 \
    --task levenshtein_prediction \
    --reset-optimizer \
    --reset-dataloader \
    --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 \
    --separator-token 2 \
    --arch roberta_base \
    --criterion levenshtein_prediction \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay \
    --lr $LR \
    --total-num-update $TOTAL_NUM_UPDATES \
    --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy \
    --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --update-freq 4  \
    --fp16 \
    --fp16-init-scale 4 \
    --threshold-loss-scale 1 \
    --fp16-scale-window 128 \
    --no-epoch-checkpoints \
    --save-dir "$OUTPUT_PATH/checkpoints" \
    --tensorboard-logdir "$OUTPUT_PATH/tensorboard"  \
    --delta-x-loss  | tee -a "$OUTPUT_PATH/train.log"


    # --wandb-project "early"