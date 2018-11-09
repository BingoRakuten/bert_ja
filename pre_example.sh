#!/bin/bash

export BERT_BASE_DIR=/root/work/bert/jamodel
FILE=/root/work/tf_examples.tfrecord


if [ -f $FILE ]; then
    echo "FILE EXISTS"
else
    python create_pretraining_data.py \
	   --input_file=/root/work/data/splitted_1.txt \
	   --output_file=/root/work/tf_examples.tfrecord \
	   --vocab_file=$BERT_BASE_DIR/vocab.txt \
	   --do_lower_case=True \
	   --max_seq_length=128 \
	   --max_predictions_per_seq=20 \
	   --masked_lm_prob=0.15 \
	   --random_seed=12345 \
	   --dupe_factor=5
fi


python run_pretraining.py \
       --input_file=/root/work/tf_examples.tfrecord \
       --output_dir=/root/work/pretraining_output \
       --do_train=True \
       --do_eval=True \
       --bert_config_file=$BERT_BASE_DIR/bert_config.json \
       --train_batch_size=32 \
       --max_seq_length=128 \
       --max_predictions_per_seq=20 \
       --num_train_steps=20 \
       --num_warmup_steps=10 \
       --learning_rate=2e-5
