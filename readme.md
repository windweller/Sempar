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

## Usage

Currently use `train_nlc.py` or `train_engine.py`

The way to use `train_engine.py` is:

## Model

We build four types of models:
1. Seq2Seq model (RNN(S1) -> Output) or (RNN(S1) -> RNN(S2) -> Output)
2. Normal Attention model (Attention goes from S1 to S2, and encoded S2 to Output)
3. Coattention model
4. Concatenated Attention Decoder Model (without multihead attention)
5. Concatenated Multi-head Attention Decoder Model (Transformer) (not yet implemented)

## Task: RNN_Engine

We directly predict the output of a query from the context.


| Model Type    | EM            | F1    | param_size |
| ------------- |:-------------:| :-----:| :-----: |
| null hypothesis (no query) | 21.76   |  82.36  | 1.84M |
| seq      |   59.91    |  94.27   | 2.63M |
| attn     |   2.65    |  24.42   | 1.97M |
| concat-attn  |   **64.17**    |  93.99  | 2.63M |
| co-attn      |   55.74    |  92.26   | 3.41M |

All models report their best EM/F1 under optimal settings.

- Null hypothesis: size 256, 20 epochs
- Seq: size 256, 20 epochs
- Attn: size 256, 20 epochs
- concat-attn: size 256, 20 epochs
- co-attn: size 256, 20 epochs

(note that concat-attn and seq have the same amount of parameters, and share basic architecture)

(note that co-attn could be under-trained because the parameter size, but size=256 outperforms size=128,
could try size=175)


## Task: RNN_Logic

We generate the logical form conditioned not just on the input query, but on
the context as well. Q2L means "Query to Logical parse"

| Model Type    | EM            | F1    | param_size |
| ------------- |:-------------:| :-----:| :-----: |
| no context (Q2L) | 55.58   |  92.72  | 1.84M |
| seq      |   **59.91**    |  94.27   | 2.63M |
| attn     |   xxx    |  xxx   | xxx  |
| concat-attn  |   49.47    |   91.88    | 2.63M |
| co-attn      |   51.48    |  92.08   | 3.42M |

All models report their best EM/F1 under optimal settings.

- no context (Q2L): size 256, 10 epochs (20 epochs running...)
- Seq: size 256, 15 epochs
- Attn: size 256, 20 epochs
- concat-attn: 256, 25 epochs
- co-attn: 256, 35 epochs

