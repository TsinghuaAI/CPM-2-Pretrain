WORKING_DIR=/home/thu-plm/CPM-2/

python3 ${WORKING_DIR}/src/tools/preprocess_data_enc_dec.py --input ${YOUR_PATH_TO}/raw_data/wudao_corpus_raw.txt --tokenizer-path ${WORKING_DIR}/bpe_cn_en/ --output-prefix ${WORKING_DIR}/pretrain_data/wudao_corpus --workers 30
