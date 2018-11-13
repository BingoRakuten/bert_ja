# BERT experiments

original version is this: https://github.com/google-research/bert

First, I needed to apply them for Japanese sentences, so prepared a data called JAS.
And then, just replaced ```run_classifier.py``` and add some lines:

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

and then, ran it. The result:

```
eval_accuracy = 0.363
eval_loss = 1.5512451
global_step = 937
loss = 1.5512451
```

it's nice because its data size isn't so large.
The baseline model ```JAS_old/run.py``` is lower (acc: 0.333).


# vocab.txt for sentencepiece

I fixed tokenization.py for sentencepiece vocaburaly.
https://github.com/sugiyamath/bert/blob/master/tokenization.py

fixed lines: 159, 211-219

it disabled "text normalization" and "charcter based tokenization for chinese characters", because wanna increase vocabulary for Japanese language.

And then, created ```pre_example.sh``` .
https://github.com/sugiyamath/bert/blob/master/pre_example.sh

