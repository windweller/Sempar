## Implementation

`train.py` is the main training file. The architecture is very similar to my CS224N's Reading Comprehension code.

 `python train.py` will launch the training, can finish in 3-4 minutes.

 `data` folder has processed queries. Naturalization brings a lot of unbalance to
 the task, and Seq2Seq model doesn't handle unbalanced source/target pair well.

 In order to tackle unbalancedness, we compute a simple ratio over the charatecter
 length of sequence, and exclude sequence where `target / source <= 15`

 This gives us 31994 pairs of training data, out of 36589 (87.44%). Many of these pairs have a one-to-many mapping
 (one input maps to multiple repeating parses). We only take one parse (since model can simply output a number to indicate
 how many repeating parse there should be). We call these `trimmed_q` in `data` folder.

 We construct another smaller dataset `small_q` strictly for one-to-one mapping (one input only generates
 one parse), and there are 14051 pairs.

 Both encoder and decoder have one layer, no outside embedding is used.  Decoder has attention (source-to-target attention).
  Vocabulary size is very small: 297 (this includes
 many markers like "[", "*", "/"), one can find them in `data/vocab.dat`.

 We use F1 and EM (exact match) score to evaluate success. F1 ignores order and just compare how many
 words are overlapping, EM considers whole sequence to match.

 The training loss (sequence-to-sequence loss) goes down from 10.780951 to 1.112809, but the F1 score only
 goes up slightly: 0.013 to 0.022, EM score remains 0.0.

 An example output is as follows:

```
input: d blue

decoded result: up plate w bottom c repeat call 10 of g height

ground truth result: isolate s select call adj bot this add blue here
```

## Proposed Fix

Unlike translation, language correction, dialogue agents, etc., espeically in this dataset, every word in source token
contains meaning, which is difficult for Seq2Seq to capture.
However, this fits into the regime of one-shot learning.

Even though there has been several papers using network with explicit memory structure on general semantic
parsing tasks (like WikiTable), but not on this dataset. And there are reasons to believe this dataset (since it's very large)
is more suitable for neural net than WikiTable.

One-shot Learning with Memory-Augmented Neural Networks
https://arxiv.org/pdf/1605.06065.pdf