1. results_analysis.md

This should be the main detailed analysis of your experimental results.

It should contain:

1. Experimental Models

Explain the three models:

1. Mean Pooling Ablation
2. Standard Additive Attention
3. Quantum-Inspired Attention
2. Performance Comparison

Use this table:

Model	Accuracy	Precision	Recall	F1-score
Mean Pooling	86.35%	84.81%	88.56%	86.65%
Standard Additive Attention	86.74%	84.80%	89.53%	87.10%
Quantum-Inspired Attention	86.37%	85.40%	87.74%	86.56%

Then explain:

Standard Attention has the highest accuracy.
Standard Attention has the highest recall.
Standard Attention has the highest F1-score.
Quantum-Inspired Attention has the highest precision.
Mean Pooling is surprisingly competitive despite having no explicit attention mechanism.
3. Error Analysis

Discuss:

Model	TN	FP	FN	TP
Mean Pooling	10518	1982	1430	11070
Standard Attention	10494	2006	1309	11191
Quantum-Inspired Attention	10625	1875	1532	10968

Explain:

Quantum-Inspired Attention produces the lowest number of false positives, which is consistent with its highest precision. Standard Attention produces the lowest number of false negatives, which contributes to its highest recall.

4. Computational Analysis
Model	Parameters	Avg. inference/sample
Mean Pooling	2,839,746	2.410 ms
Standard Attention	2,856,259	3.320 ms
Quantum-Inspired Attention	2,856,259	3.356 ms

Explain that Mean Pooling is computationally cheaper, while both attention models have the same parameter count.

5. Statistical Analysis

Discuss your existing Standard-vs-Quantum result:

McNemar p-value = 0.009969
Bootstrap 95% CI = [-0.0062, -0.0009]

Then state clearly:

The statistical analysis was performed for the paired predictions of Standard Additive Attention and Quantum-Inspired Attention. The result indicates a statistically significant difference between their predictions at α = 0.05. The negative confidence interval indicates that the observed accuracy difference favored Standard Attention.

Do not claim this statistical test includes Mean Pooling.