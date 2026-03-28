"""
gcn_numpy.py
------------
Educational implementation of a two-layer Graph Convolutional Network (GCN)
built from scratch in NumPy.

This file is NOT the production classifier used in this pipeline.
The working classifier is the gradient-boosted ensemble in 04_train_evaluate.py,
which operates on hand-engineered graph-level features extracted from
ComBat-harmonized connectomes.

Purpose of this file
--------------------
This implementation exists to make the graph convolution mathematics transparent
and inspectable without requiring PyTorch or a GPU. It demonstrates:

  - Symmetric normalized adjacency: A_hat = D^(-1/2) (A + I) D^(-1/2)
  - Two-layer GCN forward pass: H = softmax(A_hat * ReLU(A_hat * X * W1) * W2)
  - Global mean pooling: graph-level embedding from node embeddings
  - Adam optimizer with gradient clipping
  - Binary cross-entropy loss

Why it is not used as the main classifier
------------------------------------------
Graph neural networks require automatic differentiation (autograd) to compute
gradients through the adjacency normalization and multi-layer propagation steps
accurately. A numpy implementation must compute these gradients analytically,
which is error-prone and numerically unstable on real fMRI connectivity data
where graph density is ~60% and edge weights span a wide range after Fisher
z-transform. In practice, the loss collapses to NaN within a few epochs without
a stable autograd framework.

For a working GCN on ABIDE, see:
  Li X, et al. (2021). BrainGNN. Medical Image Analysis, 74:102233.
  PyTorch Geometric: https://pytorch-geometric.readthedocs.io

Usage (educational only)
------------------------
  from scripts.models.gcn_numpy import GCN
  model = GCN(in_dim=5, hidden_dim=32)
  prob  = model.forward(graph)   # forward pass only; backward is unstable on real data

References
----------
Kipf TN, Welling M. (2017). Semi-supervised classification with graph
  convolutional networks. ICLR 2017. arXiv:1609.02907.
"""

import numpy as np


def sigmoid(x):
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def relu(x):
    return np.maximum(0, x)


def normalise_adjacency(edge_index, n, edge_weight=None):
    """
    Compute symmetric normalized adjacency matrix with self-loops:
    A_hat = D^(-1/2) (A + I) D^(-1/2)
    """
    A = np.zeros((n, n), dtype=np.float32)
    if edge_weight is not None:
        A[edge_index[0], edge_index[1]] = edge_weight
    else:
        A[edge_index[0], edge_index[1]] = 1.0
    A += np.eye(n, dtype=np.float32)

    degree     = A.sum(axis=1)
    d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
    D          = np.diag(d_inv_sqrt)

    return D @ A @ D


class DenseLayer:
    """Fully connected layer with Adam optimizer."""

    def __init__(self, in_dim, out_dim, seed=0):
        rng      = np.random.default_rng(seed)
        self.W   = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, out_dim)).astype(np.float32)
        self.b   = np.zeros((1, out_dim), dtype=np.float32)
        self._m  = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self._v  = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self._dW = None
        self._db = None

    def forward(self, X):
        self._X = X
        return X @ self.W + self.b

    def backward(self, dout):
        self._dW = self._X.T @ dout
        self._db = dout.sum(axis=0, keepdims=True)
        return dout @ self.W.T

    def adam_step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for p, dp, key in [(self.W, self._dW, "W"), (self.b, self._db, "b")]:
            self._m[key] = beta1 * self._m[key] + (1 - beta1) * dp
            self._v[key] = beta2 * self._v[key] + (1 - beta2) * dp ** 2
            m_hat        = self._m[key] / (1 - beta1 ** t)
            v_hat        = self._v[key] / (1 - beta2 ** t)
            p           -= lr * m_hat / (np.sqrt(v_hat) + eps)


class GCNLayer:
    """Single GCN layer: H = ReLU(A_hat * X * W)."""

    def __init__(self, in_dim, out_dim, seed=0):
        rng      = np.random.default_rng(seed)
        self.W   = rng.normal(0, np.sqrt(2.0 / in_dim), (in_dim, out_dim)).astype(np.float32)
        self.b   = np.zeros((1, out_dim), dtype=np.float32)
        self._m  = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self._v  = {"W": np.zeros_like(self.W), "b": np.zeros_like(self.b)}
        self._dW = None
        self._db = None

    def forward(self, A_hat, X):
        self._A_hat = A_hat
        self._X     = X
        self._Z     = A_hat @ X @ self.W + self.b
        return relu(self._Z)

    def backward(self, dout):
        drelu    = dout * (self._Z > 0).astype(np.float32)
        self._dW = (self._A_hat @ self._X).T @ drelu
        self._db = drelu.sum(axis=0, keepdims=True)
        return (self._A_hat.T @ drelu) @ self.W.T

    def adam_step(self, lr, t, beta1=0.9, beta2=0.999, eps=1e-8):
        for p, dp, key in [(self.W, self._dW, "W"), (self.b, self._db, "b")]:
            self._m[key] = beta1 * self._m[key] + (1 - beta1) * dp
            self._v[key] = beta2 * self._v[key] + (1 - beta2) * dp ** 2
            m_hat        = self._m[key] / (1 - beta1 ** t)
            v_hat        = self._v[key] / (1 - beta2 ** t)
            p           -= lr * m_hat / (np.sqrt(v_hat) + eps)


class GCN:
    """
    Two-layer GCN with global mean pooling for graph-level binary classification.

    NOTE: This is an educational implementation only. See module docstring
    for why it is not used as the production classifier.
    """

    def __init__(self, in_dim=5, hidden_dim=32, seed=42):
        self.gcn1  = GCNLayer(in_dim,     hidden_dim, seed)
        self.gcn2  = GCNLayer(hidden_dim, hidden_dim, seed + 1)
        self.dense = DenseLayer(hidden_dim, 1, seed + 2)
        self._t    = 0

    def forward(self, graph):
        A_hat = normalise_adjacency(
            graph["edge_index"], graph["x"].shape[0], graph["edge_weight"]
        )
        H1    = self.gcn1.forward(A_hat, graph["x"])
        H2    = self.gcn2.forward(A_hat, H1)
        pool  = H2.mean(axis=0, keepdims=True)
        logit = self.dense.forward(pool)
        prob  = float(sigmoid(logit).squeeze())

        if np.isnan(prob):
            prob = 0.5

        self._H2   = H2
        self._prob = prob
        return prob

    def backward(self, y_true, lr=1e-3):
        self._t += 1
        y   = float(y_true)
        p   = self._prob
        eps = 1e-9

        d_logit = np.array([[p - y]])
        d_pool  = self.dense.backward(d_logit)
        N       = self._H2.shape[0]
        d_H2    = np.repeat(d_pool, N, axis=0) / N
        d_H1    = self.gcn2.backward(d_H2)
        self.gcn1.backward(d_H1)

        for layer in [self.gcn1, self.gcn2, self.dense]:
            layer.adam_step(lr, self._t)

        return float(-(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

    def predict_proba(self, graph):
        return self.forward(graph)

    def predict(self, graph, threshold=0.5):
        return int(self.predict_proba(graph) >= threshold)
