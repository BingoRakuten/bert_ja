from subprocess import check_output
import os
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

def build_command(input_file, output_file, vocab_file="/root/work/bert/jamodel/vocab.txt"):
    cmd = ["python","/root/work/bert/create_pretraining_data.py",
           "--input_file={}".format(input_file),
           "--output_file={}".format(output_file),
           "--vocab_file={}".format(vocab_file),
	   "--do_lower_case=False","--max_seq_length=128","--max_predictions_per_seq=20",
	   "--masked_lm_prob=0.15","--random_seed=12345","--dupe_factor=2"]
    result = check_output(cmd)
    return result


def execute(i, input_dir, output_dir, vocab_file):
    try:
        datanum = str(i).rjust(6, '0')
        input_file = os.path.join(input_dir, "text.txt_{}".format(datanum))
        output_file = os.path.join(output_dir, "tf_examples.tf_record_{}".format(datanum))
        if os.path.exists(output_file):
            return None
        else:
            build_command(input_file, output_file, vocab_file)
            print(str(i), end=' ', flush=True)
    except:
        print("Error:"+str(i), end=' ', flush=True)


def execute_them(ds, input_dir, output_dir, vocab_file, poolsize=10):
    try:
        pool = ThreadPool(poolsize)
        func = partial(execute,
            input_dir=input_dir,
            output_dir=output_dir,
            vocab_file=vocab_file)

        pool.map(func, ds)
    except Exception as e:
        print(e)
        print("Error in pool")
        

def main(input_dir, output_dir, vocab_file, datasize=999425):
    from multiprocessing import Pool
    import numpy as np

    poolsize = 5
    targets = np.split(np.array(list(range(datasize))), 5)
    
    func = partial(execute_them,
            input_dir=input_dir,
            output_dir=output_dir,
            vocab_file=vocab_file)
    pool = Pool(5)
    pool.map(func, targets)


if __name__ == "__main__":
    main("/root/work/data/txt_data/", "/root/work/data/records", "/root/work/bert/jamodel/vocab.txt")
