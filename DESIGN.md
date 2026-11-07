# Graph, learning and artifact contracts

## Sparse graph semantics

Input features are a rectangular list of finite numeric rows, 1-128 nodes and width 1-16, coordinates [-100,100]. Input edges are an undirected binary list `[i,j]` with integer indices in range. Self loops, duplicates (including reversed duplicates), malformed pairs and booleans fail. The implementation adds exactly one self loop per node internally. Edge-list order/orientation cannot change the operator.

For neighbor sets including self, degree d_i is their size and S_ij=1/sqrt(d_i*d_j) on each neighbor edge. S is symmetric but not generally row-stochastic. `propagate` applies sparse neighbor lists to dense feature rows. Identity propagation in MLP mode ignores validated edges. Tests compare the sparse calculation to an independently constructed dense reference and a three-node path's exact normalization.

## Forward and backward

No biases. W0 is input x hidden and W1 is hidden x classes, both flattened row-major. Compute SX, pre=SX W0, H=ReLU(pre), SH=S H and logits=SH W1. Softmax is row-wise max-shifted. Cross-entropy uses log-sum-exp minus the target logit, not log of a clipped/underflowed probability. The mean is over training nodes, not all nodes.

D is zero on every unselected node and `(P-Y)/number_selected` on training rows. Then gradients are `gW1=(SH)^T D`, `dH=S^T (D W1^T)`, and `gW0=(SX)^T (dH * (pre>0))`. S^T=S for this undirected operator. At exactly zero the ReLU derivative is defined as zero; finite-difference tests avoid that nondifferentiable point.

Two propagation operations yield a two-hop context. Unlabeled/test features may affect training-node logits and gradients through the graph. Their labels cannot. Node permutation tests permute features, edges and masks together; they are numerical equivariance checks, not expressivity guarantees or test-label isolation checks by themselves.

## Fit and checkpoint selection

Labels have one entry per node: `None` or an integer class in [0,C). Training indices are a nonempty unique list; each selected node must have a label. Optional validation indices are also nonempty/unique, disjoint from training and fully labeled. The class count is supplied explicitly, never inferred from hidden labels.

Full-batch Adam: beta1=.9, beta2=.999, epsilon=1e-8 with bias correction; no regularization, dropout or gradient clipping. Updates 1-5000, learning rate [1e-6,.1], seed integer [0,2^32-1]. Hidden width 1-32 and classes 2-8. The model uses symmetric propagation in both layers or identity in both layers.

Without validation, return the last update. With validation, score initialization and every completed update by validation cross-entropy; accept strictly smaller loss only, preserving earliest ties. Validation labels never enter optimizer gradients. Return selected weights plus a report containing selected update, final update, selected validation loss, sparse training-loss history and the final unselected inference artifact. Initialization can win. The demo reports both selected and final test metrics only after all fitting and selection complete.

The demo removes every test label before fitting. A test changes non-gradient labels and checks identical gradients/fit weights; another checks every selected checkpoint against independent Adam prefix refits. Validation labels are intentionally used for checkpoint selection, so changing them can change the returned checkpoint, not the optimization trajectory.

## Artifact and requests

`graphlab_v1` has exactly `format`, `inputs`, `hidden`, `classes`, `mode`, `parameters`. Weights have exactly inputs*hidden + hidden*classes finite numeric entries. Mode is `gcn` or `mlp`; unknown keys/versions and invalid dimensions fail. It stores no graph, node identifiers, labels, optimizer state or fitted normalization. Normalization is recomputed from each supplied inference graph. Prediction can run another graph with the same feature width, but this is capability, not evidence of transfer quality. Node-index remapping remains the caller's responsibility.

CLI train JSON requires exactly features/edges/labels/train_indices/classes, plus optional validation_indices. CLI predict JSON has exactly features/edges. Invalid requests exit 2 without a traceback. Train writes an inference artifact; it is not an exact-resume checkpoint.

## Evidence boundaries

The synthetic community graph was generated using true classes, as documented. The graph-scrambled control relabels adjacency only, deliberately breaking label/feature alignment without changing graph isomorphism class or the degree multiset. This is a descriptive control, not identification of a causal mechanism. Small validation masks can be unrepresentative, illustrated by the MLP's validation/test gap. Selected and last checkpoints are both retained to prevent hiding that behavior.

For the expressivity counterexample, each graph is degree-2 regular. With self loops S rows sum to 1; identical constant features remain constant after S, so shared nodewise transforms produce identical outputs. This is directly proved for this architecture without relying on a numerical near-equality. The fixture's maximum difference is exactly zero. No graph-level pooling is implemented.


## Fixed-model margin derivative

For target node v and distinct classes a,b, the score is logits[v,a]-logits[v,b]. It is not softmax probability or loss. Initialize dSH[v,h]=W1[h,a]-W1[h,b] and all other rows zero. Then dH=S^T dSH, dPre=dH*(pre>0), dSX=dPre W0^T and dX=S^T dSX. S is symmetric in GCN mode and identity in MLP mode. The derivative holds graph edges, normalization and learned weights fixed. Invalid target node/classes or nonfinite outputs fail.

The returned matrix has one derivative for every input coordinate. Gradient-times-input preserves each sign and per-node sum. Propagation hops are enumerated only up to 2 through the actual operator; None means outside that field, not necessarily disconnected. MLP mode's field contains only its target. Graphs with disconnected components may have a smaller field; ReLU inactivity can zero additional gradients inside the field. A zero derivative alone is not an absent finite-change effect.

With no biases and one ReLU, along the positive ray alpha*X the preactivations scale by alpha without changing the active set. The target logit margin also scales by alpha and is zero at X=0. Alternatively, expand the score as a sum of active preactivations times output coefficients: multiplying its input derivative by X reproduces each active preactivation's value. Thus sum(X*dScore/dX)=score for this architecture, including a zero-valued inactive/kink unit under the chosen zero derivative. This is a direct algebraic audit, not a new method, causal attribution, conditional feature-distribution statement or certificate that the zero reference is meaningful.

The demo uses disclosed hand-set weights, not trained benchmark weights, and retains a negative coordinate contribution. A dead-ReLU fixture distinguishes local derivatives from arbitrary replacement effects. Finite differences validate all input coordinates on tiny GCN/MLP references away from kinks. Permutation tests remap both target and graph; margin reversal negates every derivative. No edge gradients, parameter gradients of an explanation, integrated-gradients quadrature or nonzero reference baseline are offered.


## Single-edge deletion sensitivity

`edge_sensitivity` validates graph/targets and the integer budget before invoking the model. It needs E+1 forwards: the original graph and one graph per single original edge deleted. It does not mutate features, weights or edges. Budget failure is preflight, not a partial run. Return all canonical endpoint pairs with original input positions, deleted margin and signed change (deleted minus original). Empty graphs need one call. MLP mode still runs the same full audit and returns zero effects for all valid edge deletions.

Each deletion constructs the normalized graph operator anew. Frozen normalization would answer a different question and is not implemented. Because degree factors appear in the two-hop weights, an edge from a two-hop input node to a three-hop node can change a target score even though the three-hop node's own feature derivative is zero. The path counterexample computes that difference explicitly. This does not contradict fixed-graph derivative locality. Multiple deletions are not additive and no joint effect, causal interpretation, robustness certificate or generic irrelevance test is implied.
