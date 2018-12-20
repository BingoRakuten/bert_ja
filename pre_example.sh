#!/bin/bash

export BERT_BASE_DIR=/home/ubuntu/songyang/google/bert_ja/jamodel
FILE=/home/ubuntu/songyang/google/bert_ja/data/records/tf_examples.tf_record_0000*


#if [ -f $FILE ]; then
#    echo "FILE EXISTS"
#else
#    python create_pretraining_data.py \
#	   --input_file=/root/work/data/splitted_sp.txt \
#	   --output_file=/root/work/tf_examples.tfrecord \
#	   --vocab_file=$BERT_BASE_DIR/vocab.txt \
#	   --do_lower_case=True \
#	   --max_seq_length=128 \
#	   --max_predictions_per_seq=20 \
#	   --masked_lm_prob=0.15 \
#	   --random_seed=12345 \
#	   --dupe_factor=5
#fi


python run_pretraining.py \
       --input_file=$FILE \
       --output_dir=/home/ubuntu/songyang/google/bert_ja/pretraining_output \
       --do_train=True \
       --do_eval=True \
       --bert_config_file=$BERT_BASE_DIR/bert_config.json \
       --train_batch_size=32 \
       --max_seq_length=128 \
       --max_predictions_per_seq=20 \
       --num_train_steps=15000 \
       --num_warmup_steps=10 \
       --learning_rate=2e-5
