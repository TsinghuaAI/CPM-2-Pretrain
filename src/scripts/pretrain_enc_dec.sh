#! /bin/bash


WORKING_DIR=/home/thu-plm/CPM-2/

python ${WORKING_DIR}/src/libbase/setup.py install

# Change for multinode config
MP_SIZE=8

DATA_PATH="${WORKING_DIR}/pretrain_data/wudao_corpus"

CONFIG_PATH="${WORKING_DIR}/src/configs/model/enc_dec_xlarge_config.json"
CKPT_PATH=""

SAVE_PATH="${WORKING_DIR}/results/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/ds_cpm2.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_cn_en"　# 中英混合词表
# TOKENIZER_PATH="${WORKING_DIR}/bpe_cn" # 中文词表

BATCH_SIZE=1
LR=0.0001
TRAIN_ITER=12800000
LR_ITER=12800000

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
OPTS+=" --gradient-accumulation-steps 64"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --data-impl mmap"
OPTS+=" --split 949,50,1"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup 0.01"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --save-interval 50000"
OPTS+=" --eval-interval 50000"
OPTS+=" --eval-iters 10"
OPTS+=" --log-interval 100"
OPTS+=" --fp16"

CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE} ${WORKING_DIR}/src/pretrain_enc_dec.py $@ ${OPTS}"

echo ${CMD}
${CMD}

set +x
