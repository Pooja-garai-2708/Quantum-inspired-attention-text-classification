This file should explain the experimental conditions/configuration under which your conclusions were obtained.

Write things like:

Experimental Condition

Dataset

IMDB sentiment classification dataset
Binary classification
25,000 test samples

Preprocessing

Vocabulary size: 20,000
Maximum sequence length: 250
Padding/truncation: post
OOV token handling

Model Architecture

Embedding dimension: 128
BiLSTM units: 128
Classifier dense units: 64
Dropout: 0.30
Recurrent dropout: 0.20

Training

Training samples: 22,500
Validation samples: 2,500
Batch size: 64
Epochs: 2
Learning rate: 0.001
Random seed: 42

Attention Configuration

Standard:

$$ e_i=v^T\tanh(Wh_i+b) $$

followed by Softmax normalization.

Quantum-Inspired:

$$ a_i=\tanh(e_i) $$ $$ p_i=|a_i|^2 $$ $$ \alpha_i= \frac{p_i} {\sum_jp_j+\epsilon} $$

with padding masking.

Evaluation

All three models were evaluated on the same 25,000 test samples using:

Accuracy
Precision
Recall
F1-score
Confusion matrix
Inference time
Parameter count