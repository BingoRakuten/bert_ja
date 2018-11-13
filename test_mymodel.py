# coding: utf-8
from subprocess import check_output

tasks = ["JAS"]

for task in tasks:
    cmd = ["python", "run_classifier.py",
           "--task_name={}".format(task),
           "--do_train=true",
           "--do_eval=true",
           "--data_dir=/root/work/glue_data/{}".format(task),
           "--vocab_file=/root/work/bert/jamodel/vocab.txt",
           "--bert_config_file=/root/work/bert/jamodel/bert_config.json",
           "--init_checkpoint=/root/work/pretraining_output/model.ckpt-15000",
           "--max_seq_length=128",
           "--train_batch_size=32",
           "--learning_rate=2e-5",
           "--num_train_epochs=3.0",
           "--output_dir=/root/work/tasks/{}_output/".format(task.lower())]
    print(check_output(cmd))

