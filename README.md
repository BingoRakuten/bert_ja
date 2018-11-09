# BERT experiments

original version is this: https://github.com/google-research/bert

First, I needed to apply them for Japanese sentences, so I prepared a data called JAS.
And then, I just replaced ```run_classifier.py``` and add some lines, like this:

```python
class JasProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def read_tsv(self, path):
    df = pd.read_csv(path, sep="\t")
    return [(str(text), str(label)) for text,label in zip(df['text'], df['label'])]

  
  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self.read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
      self.read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1", "2", "3", "4", "5"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[0])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples
```


and then, ran it.

The result is this:

```
eval_accuracy = 0.363
eval_loss = 1.5512451
global_step = 937
loss = 1.5512451
```

it's not so good, I think. I don't have baseline model about that, but char-based 1D CNN was better.
https://qiita.com/sugiyamath/items/7cabef39390c4a07e4d8


# vocab.txt for MeCab

I wanted to create vocabulary for MeCab, so fixed tokenization.py
https://github.com/sugiyamath/bert/blob/master/tokenization.py

fixed lines: 159, 211-219

I disabled "text normalization" and "charcter based tokenization for chinese characters", because I wanted to increase vocabulary for Japanese language.

And then, I created ```pre_example.sh``` and ran it.
https://github.com/sugiyamath/bert/blob/master/pre_example.sh

create_pretraining_data.py seemed to success, but run_pretraining.py was failed because of OOM.
So that, I reduced model's network size like this:

```python
{
  "attention_probs_dropout_prob": 0.1,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "max_position_embeddings": 512,
  "num_attention_heads": 6,
  "num_hidden_layers": 6,
  "type_vocab_size": 2,
  "vocab_size": 1710509
}
```

and ran again.

Finally, my computer freezed. HELP ME!