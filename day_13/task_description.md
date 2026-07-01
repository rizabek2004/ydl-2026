# Task Description: Learning a Text Embedding Model for Job Descriptions

## Context

With only 2403 samples, you're in a small-data regime. A BiLSTM can work as a
learning project, but you need to be careful to avoid overfitting.

## Complete Architecture

Suppose your descriptions are tokenized into words:

```text
"Develop machine learning models using Python"
        ↓
["develop", "machine", "learning", "models", "using", "python"]
        ↓
[15, 832, 194, 421, 67, 98]
```

### 1. Embedding Layer

Use pretrained FastText embeddings:

```python
nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=300,
    padding_idx=0
)
```

Shape:

```text
(batch_size, seq_len)
        ↓
(batch_size, seq_len, 300)
```

Initialize with pretrained FastText and allow fine-tuning:

```python
embedding.weight.requires_grad = True
```

### 2. BiLSTM Encoder

```python
nn.LSTM(
    input_size=300,
    hidden_size=256,
    num_layers=2,
    batch_first=True,
    bidirectional=True,
    dropout=0.3
)
```

Shape:

```text
(batch, seq_len, 300)
        ↓
(batch, seq_len, 512)
```

because:

```text
256 forward + 256 backward = 512
```

### 3. Mean Pooling

The LSTM outputs a hidden state for every token:

```text
h1, h2, h3, ..., hT
```

Compute:

$$h = \frac{1}{T}\sum_{t=1}^{T} h_t$$

In PyTorch:

```python
h = outputs.mean(dim=1)
```

Shape:

```text
(batch, 512)
```

### 4. Projection Head

```python
nn.Sequential(
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 128)
)
```

Shape:

```text
(batch, 512)
        ↓
(batch, 128)
```

### 5. Normalize

```python
embedding = F.normalize(embedding, p=2, dim=1)
```

Now every description becomes:

```text
(batch, 128)
```

This is your final embedding.

## Complete Pipeline

```text
Tokens
   ↓
FastText (300)
   ↓
2-layer BiLSTM (256 hidden)
   ↓
Mean Pooling
   ↓
Linear(512 → 256)
   ↓
ReLU
   ↓
Dropout
   ↓
Linear(256 → 128)
   ↓
L2 normalization
   ↓
Embedding vector
```

## Training

But how do you train it? This is actually the difficult part.

You have:

```text
job_title → job_description
```

Since titles repeat, create **positive pairs**:

```text
Data Scientist description A
Data Scientist description B
```

and **negatives**:

```text
Data Scientist description
Software Engineer description
```

Then train with a metric-learning loss such as:

- Triplet Loss, or
- Multiple Negatives Ranking Loss

## Dataset Size Concern

2403 descriptions is tiny. Deep learning models usually like:

```text
10,000+
50,000+
100,000+
```

samples.

With only 2403 samples:

- a huge BiLSTM will overfit;
- you should use a relatively small model.

Reduce the hidden size:

```python
hidden_size = 128   # instead of 256
```

Then:

```text
BiLSTM output size = 256
```

which is plenty.

## Training Time

This depends entirely on hardware.

Assume:

- 2403 descriptions
- average length: 100 tokens
- hidden size: 128
- batch size: 32

Training for 20 epochs:

| Hardware                                  | Estimated time      |
| ----------------------------------------- | ------------------- |
| Modern GPU (RTX 3060 / M1 Pro / M2 Pro)   | 10 seconds – 2 min  |
| CPU                                       | 2 – 10 minutes      |
| MacBook with MPS acceleration             | 20 – 60 seconds     |

(GPU timing depends on implementation.)

## Overfitting (More Important Than Training Time)

With 2403 samples, use:

```python
batch_size = 32
lr = 1e-3
weight_decay = 1e-5
dropout = 0.3
epochs = 20-50
early_stopping = True
```

## Open Concern: Are There Enough Positive Pairs?

There may not be enough repeated titles to form many positive pairs. For example:

```text
Data Scientist    : 2 descriptions
ML Engineer       : 1 description
Backend Developer : 1 description
...
```

If most titles appear only once, metric learning becomes difficult because there
aren't enough positive examples. In that case, the BiLSTM approach will struggle.

## Questions to Determine Feasibility

1. How many unique job titles are there?
2. On average, how many descriptions belong to each title?

That information determines whether the metric-learning approach is feasible or
whether to switch to a pretrained sentence embedding model.
