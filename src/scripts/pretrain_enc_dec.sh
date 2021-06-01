#! /bin/bash

WORKING_DIR=/home/thu-plm/CPM-2/

# Change for multinode config
MP_SIZE=4

DATA_PATH="${WORKING_DIR}/pretrain_data/wudao_corpus"

CONFIG_PATH="${WORKING_DIR}/src/configs/model/enc_dec_xlarge_config.json"
CKPT_PATH=""

SAVE_PATH="${WORKING_DIR}/results/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/ds_cpm2.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn_en"
# TOKENIZER_PATH="${WORKING_DIR}/bpe_cn"

BATCH_SIZE=26
LR=0.001
TRAIN_ITER=200000

ENC_LEN=512
DEC_LEN=256

NUM_WORKERS=40
NUM_GPUS_PER_WORKER=8
HOST_FILE="${WORKING_DIR}/src/configs/host_files/hostfile-cpm2"

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
# OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-impl mmap"
OPTS+=" --lazy-loader"
OPTS+=" --tokenizer-type GPT2BPETokenizer"
OPTS+=" --split 949,50,1"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.05"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 4000"
OPTS+=" --eval-interval 2000"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 100"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"

CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/src/pretrain_enc_dec.py $@ ${OPTS}"

echo ${CMD}
${CMD}

set +x
