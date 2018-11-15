# coding: utf-8
with open("tmp.txt") as f:
    data = [x.strip() for x in f]
    
len(data)
data[0][:10]
data[1][:10]
data[2][:10]
data[3][:10]
data[4][:10]
data = [x for x in data if x and x.count("。") > 1 ]
len(data)
for i, d in enumerate(data):
    with open("text.txt_{}".format(str(i).rjust(6, '0')), "w"):
        pass
    
import sentencepiece as sp
import sentencepiece as sp
spm = sp.SentencePieceProcessor()
spm.Load("../bert/jamodel/jawiki.model")
from tqdm import tqdm
for i, d in tqdm(enumerate(data)):
    with open("text.txt_{}".format(str(i).rjust(6, '0')), "w") as f:
        d = [' '.join(spm.EncodeAsPieces(x)) for x in d.split("。")]
        f.write('\n'.join(d))
        
get_ipython().system('pip install tqdm')
get_ipython().run_line_magic('mkdir', 'txt_data')
from tqdm import tqdm
for i, d in tqdm(enumerate(data)):
    with open("txt_data/text.txt_{}".format(str(i).rjust(6, '0')), "w") as f:
        d = [' '.join(spm.EncodeAsPieces(x)) for x in d.split("。")]
        f.write('\n'.join(d))
        
get_ipython().run_line_magic('save', '1-19 separate_data.py')
