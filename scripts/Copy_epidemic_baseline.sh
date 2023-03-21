#!/bin/bash
if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

LANG=el
echo "Run on GPU $GPUID"
echo "language: $LANG"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
## DATA_ROOT=$PROJECT_ROOT/dataset/conll03_distant/

DATA_ROOT=$PROJECT_ROOT/dataset/epidemic/$LANG
# DATA_ROOT=$PROJECT_ROOT/dataset/epidemic/relevant/xlm_128


rm -rf $PROJECT_ROOT/outputs/*
rm -rf $PROJECT_ROOT/dataset/epidemic/$LANG/cached_*
rm -rf $PROJECT_ROOT/runs/*


MODEL_TYPE=xlmroberta
MODEL_NAME=xlm-roberta-base

# MODEL_TYPE=bert
# MODEL_NAME=bert-base-multilingual-cased

# params
LR=1e-5
WEIGHT_DECAY=1e-4
EPOCH=3
SEED=42

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=200

TRAIN_BATCH=16
EVAL_BATCH=16

# output
OUTPUT=$PROJECT_ROOT/outputs/epidemic/baseline/${MODEL_TYPE}_${EPOCH}_${LR}/

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 run_ner.py --data_dir $DATA_ROOT \
  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --logging_steps 200 \
  --save_steps 100000 \
  --do_train \
  --do_eval \
  --do_predict \
  --evaluate_during_training \
  --output_dir $OUTPUT \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --seed $SEED \
  --max_seq_length 256 \
  --overwrite_output_dir \
  # --do_lower_case \