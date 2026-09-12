# Quantum-Inspired Attention Mechanism for Text Classification

## Overview
This project compares Standard Additive Attention, Quantum-Inspired Attention,
and Mean Pooling for IMDB sentiment classification.

## Architecture

Embedding → BiLSTM → Attention/Mean Pooling → Dense Classifier → Softmax

## Dataset
- IMDB Dataset
- Binary sentiment classification
- Vocabulary: 20,000
- Maximum sequence length: 250

## Models
1. Standard Additive Attention
2. Quantum-Inspired Attention
3. Mean Pooling Ablation

## Final Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Standard Attention | 86.74% | 84.80% | 89.53% | 87.10% |
| Quantum-Inspired Attention | 86.37% | 85.40% | 87.74% | 86.56% |
| Mean Pooling | 86.35% | 84.81% | 88.56% | 86.65% |

## Conclusion
Standard Additive Attention achieved the best overall performance.
Quantum-Inspired Attention achieved the highest precision but did not
outperform Standard Attention overall.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt