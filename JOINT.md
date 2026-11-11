# Joint deletion audit: one edge at a time can miss the change

## Contract

For a fixed loaded two-layer model, feature matrix X, original undirected edge list E, target node and two different classes, enumerate every subset D of E with cardinality at most k. The empty set is included. Each row calls the original forward model on E minus D with self loops re-added and every degree recomputed. No weights, features or labels change. Model inference never accepts labels.

Rows are ordered by cardinality, then lexicographic original edge indices. Edge endpoints are displayed sorted, but IDs refer to the caller's original order. Reversing an edge orientation does not change the graph. Reordering input edges changes IDs and deterministic ordering, not the set of perturbed graphs. Argmin/argmax summary ties retain the first row, using the actually evaluated binary64 values without a tolerance-based tie merger.

The score is class-positive minus class-negative **logit**, not probability or correctness. All target logits are retained separately. Prediction takes the largest logit among **every** class, with lowest class index winning exact evaluated ties. `prediction_changed` compares against this model's baseline prediction, not a known true label. A negative chosen two-class margin need not change an all-class prediction.

## Count and budget before inference

Required calls = sum(comb(m,j), j=0..min(k,m)), where m is the number of original edges. Native limits: original 1..128 nodes, 1..16 features, hidden width 1..32, 2..8 classes; k is an exact integer in 0..8; forward budget 1..65,536. A request above the cap fails before any forward call, rather than sampling or returning a partial list. k greater than m includes all possible subsets once. Zero edges needs one call. These are call bounds, not runtime or memory promises. A large model with many allowed calls may still be slow.

The page adds practical limits of 12 nodes and 12 edges, four features, four hidden units, four classes and 4,096 calls. All entered weights/features are bounded only as the original model contract states: finite weights and features in [-100,100]. Nonfinite computed margins, changes, sums or residuals are rejected. Boolean values are not integers. Undirected duplicate edges and caller-supplied self loops are rejected.

## Nonadditivity, explicitly defined

For score f(D), delta(D) = f(D)-f(empty). The singleton extrapolation is sum(delta({e}), e in D). The residual is delta(D) minus this sum. It is zero by construction for empty/singleton rows. It is a finite functional comparison, **not** a causal interaction, statistical interaction estimate, Shapley value or derivative. Even the sign of this residual need not be stable across different models or inputs. Original single-edge and input-gradient methods stay separate.

The hand-set four-node control uses features [-4,-4,6,6], edges e0=(0,1), e1=(0,2), e2=(0,3), e3=(1,2), e4=(1,3), two ReLU hidden channels with input weights [1,-1], and output columns [0,1] / [0,-1]. Target node 0, class 1 minus class 0. This is an arithmetic control, not a trained-network performance result. It was found by enumerating four integer feature values on a fixed graph and selecting the first strong nonadditivity control; it is not an untouched statistical sample.

Baseline margin is 0.5534180126. All five single deletions preserve class 1. Deleting e1 and e3 gives margin -0.6666666667 and class 0. Summing those singleton effects would predict margin +0.3151513784, still class 1; actual residual is -0.9818180451. Of the 16 empty/single/pair sets, five change the prediction. The overall minimum margin is -1.0606601718 at e1/e2, a different pair. The interface does not conceal this stronger worst case in order to favor the selected explanatory pair.

## Checks and what they establish

- 51 native tests retain all 42 existing tests. The new independent dense reference uses full adjacency matrices, separate matrix products and bitmask subset enumeration for 180 seeded small graph/model cases. It checks every admissible subset, order, logits, class and budget summaries. This is not an exhaustive class of all feature values or all accepted graphs.
- A 70-digit Decimal square-root/matrix implementation checks all 16 four-node perturbation margins, to an explicit 3e-15 bound. This finite high-precision reference does not certify every supported input or all binary64 rounding.
- Separate JavaScript dense inference and recursive combination generation match 46 complete Python reports containing 559 subsets. Floating values use a stated 2e-11 relative/absolute scale; indices, model structure and booleans are discrete fields. No cross-language exact tie-equivalence claim is made outside the observed controls.
- Actual NumPy 2.2.6 matrix multiplication independently checks all 559 target logit vectors; maximum observed absolute difference is 8.881784197001252e-16. Discrete predictions are checked away from a 1e-10 numerical tie or for exact all-zero ties. This is numerical reference evidence, not a formal verifier or benchmark.
- Five Node/actual-Chrome tests cover 24 control-width views, 568 complete subset selections and table rows over all budgets, six trusted report downloads replayed by native sparse Python, trusted input/compute and invalidation. Widths are 1280,768,390,320. The no-edge, zero-tie, multiclass, MLP and normalization-path controls remain visible.
- The installed wheel imports seven modules from site-packages outside checkout and exercises all nine commands through both module and console entrypoints, 18 invocations. Original teacher-free synthetic fits, figures, fixed-mask behavior and the original three-arm development results reproduce unchanged.

The new browser report has `graphlab-joint/1` schema; existing inference artifacts stay `graphlab_v1`. No original stored model is rewritten. Downloaded `{input,report}` files are unsigned evidence bundles. `examples/replay_joint.py` evaluates the embedded input using native sparse inference and compares every field. This checks consistency with embedded input, **not** source identity, authentication, real-world truth or a formal proof. Changing an input and recomputing its report is expected to yield another valid bundle. The old CLI JSON loader does not reject duplicate keys; the page uses ordinary JSON parsing, with the same limitation.

## Reproduce

```sh
python -m unittest discover -s tests -v
PYTHONPATH=. python examples/export_joint.py --check
node --test tests/*.cjs
PYTHONPATH=. python examples/reference_joint.py  # optional numpy==2.2.6
python examples/check_installed.py
python -m graphlab joint-demo > /tmp/joint.json
python -m graphlab joint model.json input.json --node 0 --positive 1 --negative 0 --max-deletions 2 --max-forwards 4096
PYTHONPATH=. python examples/replay_joint.py downloaded.json
```

The page runs independent JavaScript on disclosed hand-set controls, not the original trained development graph. It accepts small custom models and graphs. The Python API accepts loaded trained models within its preflight bounds. Original transductive test accuracy does not become an adversarial accuracy result, and no retraining or adversarial training was performed. The original 483-edge graph at k=2 exceeds the native cap; k=1 fits. Do not silently reduce its candidate edge family and call that a full audit.

## Scope and sources

Original edges only. No additions, rewiring, feature attacks, topology uncertainty, semantic/causal edge deletion, temporal or unseen-graph generalization, adaptive robustness comparison, formal real-arithmetic certificate, scalable solver or new algorithm. Existing graph verification and stability research already studies these questions at greater scope. Our contribution is an original small, transparent implementation and its disclosed controls. No third-party code, data, weights or figures copied. MIT code; synthetic data only.

1. Kipf and Welling (2017), [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/html/1609.02907v4), equations 2/9: symmetric normalized propagation, self loops, two-layer architecture. Original model credit retained.
2. Chen et al. (IJCAI 2021), [Understanding Structural Vulnerability in Graph Convolutional Networks](https://ijcai.org/proceedings/2021/310): structural perturbations and aggregation vulnerability are established work. No reproduction of their results.
3. Liu, Lu and Kwiatkowska (2025), [Exact Verification of Graph Neural Networks with Incremental Constraint Solving](https://arxiv.org/html/2508.09320v1): existing formal constraint-based verification under specified edge/attribute perturbations. This census is not their solver or a claim of equivalent guarantees.
4. Geisler et al., [Adversarial Training for Graph Neural Networks](https://arxiv.org/html/2306.15427v1), section on realistic perturbations: global edge counts alone need not represent plausible local attacks. No realism guarantee is inferred from a small deletion budget.
5. Kenlay, Thanou and Dong, [On the Stability of Graph Convolutional Neural Networks under Edge Rewiring](https://arxiv.org/abs/2010.13747): established structural-stability analysis. Abstract inspected; no bound from the paper implemented or claimed.
6. Python, [itertools documentation](https://docs.python.org/3/library/itertools.html): combination ordering and finite subset generation.
7. NumPy, [matmul documentation](https://numpy.org/doc/stable/reference/generated/numpy.matmul.html): independent dense reference semantics. NumPy is optional checking software, not a runtime dependency.
