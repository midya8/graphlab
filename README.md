# GraphLab

**A graph neural network whose context, gradients and blind spots are visible.** Dependency-free two-layer graph convolution, masked node supervision, validation-only checkpoint selection and matched feature/topology controls.

![Synthetic graph and expressivity audit](assets/graph-audit.svg)

## Run offline

```sh
python -m unittest discover -s tests -v
python -m graphlab demo > /tmp/graph-demo.json
python examples/render_demo.py /tmp/graph-demo.json --check
python -m graphlab train examples/train.json /tmp/model.json --updates 100
python -m graphlab predict /tmp/model.json examples/predict.json
```

Python 3.10+, standard library only. No downloaded graph, pretrained model, account, GPU, external dependency or paid API. The full demo takes about seven seconds on the development machine; runtime depends on hardware/Python.

## The model

For undirected binary adjacency A and added self loops:

```
S = D^(-1/2) (A + I) D^(-1/2)
H = ReLU(S X W0)
P = softmax(S H W1)
```

Sparse neighbor lists apply the symmetric normalized operator. It is **not row normalization**; irregular-graph row sums need not equal one. Isolated nodes retain their own features through a self loop. A feature-only MLP uses identity propagation with the same parameterization, initialization, parameter count and update budget.

Cross-entropy gradients use training nodes only. Validation labels can select a checkpoint, including initialization, but never enter the training gradient. Test labels are passed as `None` during fitting. Test **features and graph edges remain visible**: this is transductive node evaluation on one graph, not unseen-graph, unseen-node or temporal generalization. Prediction accepts only features/edges, never labels.

[DESIGN.md](DESIGN.md) records equations, masks, selection, bounds and strict artifacts.

## Measured development fixture

Seed 41: 96 nodes in two equal class communities, 4 noisy features, 483 edges. Each coordinate has class mean -0.25 or +0.25, noise SD 1. Edges are sampled with probability 0.2 within a class and 0.015 across classes. Thus **graph construction deliberately carries class signal**. Split: 8 training + 8 validation + 32 test nodes per class. The complete node lists are in demo JSON.

Each neural arm has 72 weights, shared seed 13, Adam learning rate 0.03 and 300 updates. Validation cross-entropy selects each arm independently among initialization and every update, with earliest ties retained. No test labels select a checkpoint. The scrambled graph uses seed 47 to relabel the same graph while keeping features/labels fixed: it preserves the graph's degree multiset and structure, not each original node's degree or feature alignment.

| Neural arm | Selected update | Selected train acc. | Selected validation acc. | Selected test acc. | Selected test CE | Last-update test acc. / CE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GCN | 300 | 1.0000 | 1.0000 | 0.96875 | 0.060893 | 0.96875 / 0.060893 |
| Feature-only MLP | 1 | 0.6250 | 0.9375 | 0.34375 | 0.848734 | 0.484375 / 8.140771 |
| Scrambled-graph GCN | 0 | 0.3750 | 0.5625 | 0.515625 | 0.693328 | 0.546875 / 2.749939 |

A separate **train-only nearest-class-centroid** feature control fits 8 numbers with no validation selection and gives test accuracy **0.484375**. Balanced constant predictions give 0.5 on this balanced test set. The neural controls are not suppressed when a later checkpoint has higher test accuracy: both selected and last checkpoints are shown.

The feature MLP's excellent small-validation accuracy does not transfer to this test split; validation selection is no guarantee of representative evaluation. These are descriptive fixture results, not causal contributions or a general GCN ranking. Validation selection and the centroid control were added after inspecting the initial fixed-budget arms and their overfit losses. This is a **development fixture**, not an untouched benchmark or a multi-seed study. No architecture or seed sweep was run.

## Exact audits and an exact limitation

23 tests cover every parameter of tiny reference models by finite differences in both modes, sparse-vs-dense forward arithmetic, independent degree normalization, permutation-equivariant predictions and training, mask isolation, stable unclipped cross-entropy, selection vs independent prefix refits, artifacts and CLI.

A six-cycle and two disconnected triangles are different graphs, but with identical constant node features and degree 2, every normalized neighborhood produces the same feature at every layer. This model gives **identical node probabilities** on both. The counterexample is exact for this architecture and fixture; it is not a claim that all graph neural networks fail on those graphs. No graph-level classifier or universal graph-isomorphism test is implemented.

## Scope and limits

Small, undirected, unweighted graphs only: 1-128 nodes, 1-16 features, hidden width 1-32 and 2-8 classes. No edge features, attention, graph-level pooling, minibatching, dropout, weight decay, gradient clipping, calibration or confidence guarantee. The original paper's full training recipe and citation-network benchmarks are **not** reproduced. The figure's positions use true classes for a disclosed schematic, not a learned embedding.

Inference artifacts store weights and architecture, not graph data or optimizer state. No private/customer data is used and there is no privacy guarantee. This is an educational implementation, not a production library.

## Source credit

Independent educational implementation of **Kipf and Welling (2017), [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/html/1609.02907v4)**, especially normalized propagation Eq. 2 and the two-layer model/masked cross-entropy Eq. 9-10. Source read for mathematics; no copied code, dataset or pretrained weights. Published benchmark results belong to the paper, not this project.

MIT license.
