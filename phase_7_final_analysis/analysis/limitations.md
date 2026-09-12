This is very important for a research project.

Do not hide the fact that Quantum Attention did not win. Instead, explain the limitations honestly.

You can write:

Limitations
1. Limited training duration

The models were trained for only two epochs. Therefore, the reported results may not represent the maximum achievable performance of each architecture.

2. Single dataset

The evaluation was performed on the IMDB sentiment classification dataset. Therefore, the generalization of the findings to other NLP tasks and datasets remains uncertain.

3. Quantum-inspired formulation

The proposed mechanism is quantum-probability-inspired, rather than a quantum neural network implemented on quantum hardware. The probability formulation is based on amplitude-like values and squared magnitudes.

4. Squared amplitude loses sign information

The transformation

$$ p_i=|\tanh(e_i)|^2 $$

removes the sign of the attention score. Positive and negative scores having similar magnitudes can therefore receive similar probability values.

5. Computational evaluation

Inference time was measured on the available CPU environment. Hardware-specific differences may affect the measured execution time.

6. Statistical comparison scope

The McNemar and bootstrap analyses reported in this study compare Standard Additive Attention and Quantum-Inspired Attention. Mean Pooling was included as an ablation baseline, but the existing statistical test does not establish statistical significance involving Mean Pooling.

7. Attention interpretability

Attention visualization provides qualitative evidence about token weighting but should not automatically be interpreted as a complete explanation of model decision-making.