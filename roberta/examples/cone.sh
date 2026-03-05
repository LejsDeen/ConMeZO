#!/bin/bash

TASK=${TASK:-SST-2}
K=${K:-512}
SEED=${SEED:-100}
BS=${BS:-64}
LR=${LR:-1e-6}
EPS=${EPS:-1e-3}
WD=${WD:-0}
STEP=${STEP:-10000}
EVAL_STEP=${EVAL_STEP:-1000}
MODEL=${MODEL:-roberta-large}

CONE_THETA=${CONE_THETA:-1.35}
CONE_BETA=${CONE_BETA:-0.95}

LOGITS=$(jq -n '{"SNLI": 3, "MNLI": 3, "trec": 6, "sst-5": 5}["'$TASK'"] // 2')

GR_TAG=seed$SEED-bs$BS-lr$LR-eps$EPS-wd$WD-step$STEP-evalstep$EVAL_STEP
EXTRA_TAG=${EXTRA_TAG:-ft-}
TAG=${TAG:-k${K}-${MODEL}-dpzero-${EXTRA_TAG}}
echo "Grid search tag: $GR_TAG"
echo "Tag: $TAG"

TYPE=prompt GRID_TAG=$GR_TAG TAG=$TAG STEPS=$STEP TASK=$TASK SEED=$SEED MODEL=$MODEL K=$K \
    bash examples/run_fewshot.sh \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --eval_steps $EVAL_STEP \
    --weight_decay $WD \
    --zero_order_eps $EPS \
    --zero_order_optim \
    --lr_scheduler_type "constant" \
    --optimizer "sgd" \
    --efficient_zero_order \
    --cone \
    --cone_theta $CONE_THETA \
    --cone_beta $CONE_BETA \
    $@
