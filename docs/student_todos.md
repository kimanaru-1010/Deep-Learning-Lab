# Student TODO inventory

Không có DL core solution trong repository.

| Nhóm | Việc bạn cần tự làm |
|---|---|
| Manual | Linear forward/backward (dX/dW/db), ReLU, MSE, stable CrossEntropy, two-layer MLP và chain rule |
| Scalar | Value add/multiply/power, local derivatives, graph construction, topo traversal, accumulation |
| Tensor | Differentiable operations, indexing, matmul, sums/means, reshape/transpose, exp/log/softmax, concat/stack, unbroadcast |
| Autograd | Reverse topological traversal, upstream seeding, propagation, shared-node accumulation, requires_grad behavior |
| Dense | Linear, ReLU/Sigmoid/Tanh |
| Losses | Mean MSE, numerically stable mean CrossEntropy from logits |
| Optimizers | SGD, Momentum velocity, Adam moments/counter/bias correction/epsilon |
| Initialization | Xavier/Glorot and He/Kaiming; document fan convention |
| Vision | Conv2D, MaxPool2D forward/backward, BatchNorm train/eval, ResidualBlock |
| Sequence | Embedding repeated-index gradients, RNNCell/RNN/BPTT, LSTMCell/LSTM and both state paths |
| Attention | ScaledDotProductAttention, mask handling, MultiHeadAttention projections/head layout |
| Transformer | LayerNorm, positional encoding, pre-norm block, feed-forward/residual, causal TinyTransformer composition |
| Model composition | Manual MLP, residual block, RNN/LSTM language models, causal Transformer |

MLP và CNN model composition đã được nối từ các layer TODO để có experiment đầu
tiên rõ ràng. Không có layer/optimizer algorithm được viết sẵn. BPTT có thể đi qua
Tensor graph của bạn; không có generic backward ẩn trong training utility.

Đã hoàn thành: ndarray storage, Parameter/Module traversal và train/eval flags,
registered buffers, simple normal/uniform/zeros allocation, seed, DataLoader,
train/val split, MNIST/Text pipeline, finite differences, metrics, monitoring,
logging, checkpoint, plots, CLI và training control flow.

Tìm vị trí còn thiếu: `rg 'TODO\(student\)|todo\(' src models experiments`.
Chạy bài bằng `python -m pytest PATH --run-student -q`; TODO phải fail khi bạn đang
nghiệm thu bài. Không đổi tests để chấp nhận output sai.

