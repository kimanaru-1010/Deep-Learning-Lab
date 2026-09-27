# Implementation order

Hạ tầng đã có; danh sách này dành cho quá trình bạn tự học. Mỗi bước cần
known-value test và (khi khả vi) float64 gradient check trước khi ghép model.

1. Chain rule / manual derivatives.
2. Manual Linear forward/backward.
3. Manual ReLU.
4. Manual MSE/CrossEntropy và MLP.
5. Scalar Value operations.
6. Reverse topological backward trên scalar.
7. NumPy Tensor wrapper.
8. Tensor primitive ops.
9. Tensor traversal, propagation, accumulation và unbroadcast.
10. Đọc Module traversal có sẵn.
11. Đọc Parameter storage có sẵn.
12. Linear.
13. ReLU/Sigmoid/Tanh.
14. MSE/CrossEntropy.
15. SGD.
16. MLP composition.
17. MNIST tiny-batch overfit.
18. MNIST subset.
19. Full MNIST và held-out test cuối cùng.
20. Momentum.
21. Adam và optimizer comparison.
22. Conv2D.
23. MaxPool2D.
24. CNN.
25. Normalization và train/eval statistics.
26. Residual Block.
27. ResNet.
28. RNNCell.
29. RNN / BPTT.
30. LSTMCell.
31. LSTM.
32. Embedding.
33. Scaled Dot-Product Attention.
34. Multi-Head Attention.
35. LayerNorm.
36. Positional Encoding và Transformer Block.
37. Small causal Transformer training.

Không thêm model lớn hoặc image generative models. CNN/ResNet dùng MNIST; sequence
lessons dùng corpus nhỏ. Xavier/He có thể làm ngay sau MLP để so sánh activation và
gradient variance. Đọc docs/testing_strategy.md để phân biệt setup pass với lesson pass.

