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
| no context (Q2L) | 55.90   |  92.81  | 1.84M |
| seq      |   **59.91**    |  94.27   | 2.63M |
| attn     |   6.74    |  69.61   | 1.97M  |
| concat-attn  |   49.47    |   91.88    | 2.63M |
| co-attn      |   51.48    |  92.08   | 3.42M |

All models report their best EM/F1 under optimal settings.

- no context (Q2L): size 256, 20 epochs
- Seq: size 256, 15 epochs
- Attn: size 256, 20 epochs
- concat-attn: 256, 25 epochs
- co-attn: 256, 35 epochs

