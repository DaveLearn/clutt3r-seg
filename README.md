<div align="center">

# Clutt3R-Seg: Sparse-view 3D Instance Segmentation for Language-grounded Grasping in Cluttered Scenes

[Jeongho Noh](https://jeonghonoh.github.io/)<sup>1</sup>,
[Tai Hyoung Rhee](https://williamrheeth.github.io/)<sup>1</sup>,
[Eunho Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+E)<sup>1</sup>,
[Jeongyun Kim](https://jeongyun0609.github.io/)<sup>1</sup>,
[Sunwoo Lee](https://arxiv.org/search/cs?searchtype=author&query=Lee,+S)<sup>2</sup>,
[Ayoung Kim](https://ayoungk.github.io/)<sup>1,†</sup>

<sup>1</sup> Seoul National University &nbsp;&nbsp;
<sup>2</sup> Hyundai Motor Company &nbsp;&nbsp;
<sup>†</sup> Corresponding Author

**ICRA 2026**

[![arXiv](https://img.shields.io/badge/arXiv-2602.11660-b31b1b.svg)](https://arxiv.org/abs/2602.11660)

</div>

Clutt3R-Seg is a zero-shot sparse-view 3D instance segmentation pipeline that builds hierarchy-based, open-vocabulary 3D instances for language-grounded grasping in cluttered scenes.

It groups noisy RGB-D masks across views, resolves over- and under-segmentation through an instance tree, and updates object correspondences after robot interactions without rescanning the full scene.

Paper: [arXiv:2602.11660](https://arxiv.org/abs/2602.11660)

![Clutt3R-Seg pipeline](assets/pipeline.jpg)

## Clone

```bash
git clone https://github.com/jeonghonoh/clutt3r-seg
cd clutt3r-seg
```

## Build

Build the runtime image locally:

```bash
docker build --build-arg INSTALL_DUODUOCLIP=1 -t clutt3r-seg:duoduoclip-local .
```

This local build downloads and installs [DuoduoCLIP](https://github.com/3dlg-hcvc/DuoduoCLIP)
from its upstream repository inside the image. Clutt3R-Seg does not redistribute DuoduoCLIP
source code, checkpoints, or Docker images with DuoduoCLIP preinstalled.

Use of DuoduoCLIP is subject to its upstream license and dependency licenses.
Do not publish or redistribute Docker images with DuoduoCLIP preinstalled unless
you have confirmed that all relevant licenses allow it.

## Data

This release includes sample sequences from
[GraspClutter6D](https://sites.google.com/view/graspclutter6d), along with
custom real-world and synthetic sequences.

Each sequence should follow this layout:

```text
samples/<sequence_name>/
  data/
    transforms.json
    images/
    depth/
    instance_masks/
    instance_tree.json
```

`transforms.json` must contain camera intrinsics and per-frame `file_path`,
`depth_file_path`, and `transform_matrix` entries. Instance masks should be
stored as `mask_<frame_id>_<instance_id>.png`.

`data/depth` should contain dense MVSAnywhere inference depth, as used in the
paper pipeline. Raw measured depth with invalid pixels can break back-projection
and geometry consistency.

`data/instance_tree.json` stores precomputed instance-tree assignments for this
source-available release. The public release does not include the internal
instance-tree builder because parts of that implementation depend on closed or
restricted components that cannot be redistributed. The paper describes the
instance-tree construction procedure at the level intended for reproduction;
new sequences need a compatible precomputed artifact. The artifact must match
the exact `instance_masks/` files used by the run.

For update, the selected update frame must have RGB, instance masks, and
update-frame tree entries in `instance_tree.json`. Update-frame depth is only
used for optional depth evaluation when available.

Included sample sequences:

- `samples/sample_seq1`: custom real-world sequence with more than eight frames;
  supports both initial segmentation and update.
- `samples/sample_seq2`: custom real-world sequence with more than eight frames;
  supports both initial segmentation and update.
- `samples/sample_seq3`: difficult sequence from GraspClutter6D.
- `samples/sample_seq4`: easy sequence from GraspClutter6D.
- `samples/sample_seq5`: custom synthetic sequence captured in Isaac Sim.

See `samples/README.md` for additional sequence layout notes.

## Run

Initial segmentation:

```bash
docker run --rm --gpus all \
  -v "$PWD/samples:/workspace/samples" \
  -v "$PWD/.cache:/workspace/.cache" \
  clutt3r-seg:duoduoclip-local \
  bash scripts/run_initial.sh samples/sample_seq2 0,1,2,3,4,5,6,7 "cracker box"
```

Update segmentation and scene update:

```bash
docker run --rm --gpus all \
  -v "$PWD/samples:/workspace/samples" \
  -v "$PWD/.cache:/workspace/.cache" \
  clutt3r-seg:duoduoclip-local \
  bash scripts/run_update.sh samples/sample_seq2 8 1 "chips can"
```

Run update only after initial segmentation has created
`samples/<sequence_name>/state.pkl`.

Outputs are written under `samples/<sequence_name>/output/`:

- `<target_prompt>.ply`: prompt-matched target object point cloud.
- `updated_scene_<update_num>.ply`: full updated scene from the update step.

The 6-DoF grasp pose estimation stage used in the paper is not included in this
release; exported object point clouds can be used as input to external
grasp-pose estimators.

For an interactive shell:

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD/.cache:/workspace/.cache" \
  clutt3r-seg:duoduoclip-local \
  bash
```

## Instance-tree builder (reconstruction)

The public release ships `data/instance_tree.json` for each sample but not the
builder that produces it. This fork reconstructs that builder from the paper in
`clutt3rseg/tree_builder.py` (+ `clutt3rseg/scene_substrate.py`), following
Algorithm 1: per-frame 2D containment forests select proper-segment *leaf* masks
(`phi`), which form the vertices of a complete graph over **cross-frame** leaf
pairs (`phi(u) != phi(v)`); the graph is contracted greedily by **average
linkage** (`GroupAndRewire`, the new edge weight is the mean similarity over all
constituent pairs) in two stages — first by super-voxel spatial similarity
(`S_spatial >= tau_spat`), then by DuoduoCLIP semantic similarity
(`S_semantic >= tau_sem`) — followed by residual-node parent substitution.

### Building a tree

```bash
# writes <seq>/data/instance_tree.json
pixi run build_tree samples/sample_seq2 --initial-idx 0,1,2,3,4,5,6,7 --update-idx 8,9
```

`segment.py` also auto-builds the tree when it is missing
(`--build-tree-if-missing`, on by default), so new sequences no longer require a
precomputed artifact — only the Grounded-SAM `instance_masks/` and dense depth.

### Grouping variant flag

`build_tree.py` and `segment.py` take `--variant {paper,improved}`. Both use the
same Algorithm-1 machinery (complete cross-frame leaf graph, two-stage greedy
contraction, **average linkage**, residual substitution); they differ only in the
spatial-similarity term and its threshold:

| variant | spatial term | `tau_spat` | `tau_sem` | linkage |
|---|---|---|---|---|
| `paper` (default) | weighted Jaccard (∩/∪) | 0.50 | 0.65 | average |
| `improved` | weighted overlap coeff. (∩/min) | 0.40 | 0.65 | average |

`paper` is the literal method section and is the default. `improved` makes one principled change: the
spatial term divides by the smaller mask's mass instead of the union (overlap
coefficient). Rationale (see `_pair_spatial`): a partial cross-view mask of an
object overlaps only ~0.5 of the accumulated object under Jaccard and can miss
`tau_spat`, whereas distinct objects share ~0 fine 5 mm super-voxels
(inter-instance overlap p90 < 0.01), so the overlap coefficient raises recall
without hurting precision. Individual knobs override the variant: `--spatial-metric`,
`--tau-spat`, `--tau-sem`, `--linkage`, `--containment-thresh`.

### Reproducing the shipped trees

`scripts/reproduce_sample_trees.py` rebuilds every bundled sample from its raw
data and compares to the shipped artifact:

```bash
pixi run --frozen python scripts/reproduce_sample_trees.py --variants paper,improved
```

Agreement with the shipped `instance_tree.json` — Adjusted Rand Index (ARI) of the
two groupings on shared leaves, mean best-match instance IoU, instance counts
(built/shipped), and the fraction of shipped leaf masks recovered (leaf-cov):

| sequence | variant | ARI | mean IoU | inst (built/shipped) | leaf-cov |
|---|---|---|---|---|---|
| sample_seq1 | paper | 1.000 | 1.000 | 9/10 | 0.95 |
| sample_seq2 | paper | 0.883 | 0.690 | 15/10 | 0.97 |
| sample_seq3 | paper | 0.929 | 0.809 | 31/27 | 0.93 |
| sample_seq4 | paper | 0.935 | 0.943 | 13/15 | 0.91 |
| sample_seq5 | paper | 0.806 | 0.571 | 16/8 | 0.98 |
| **mean** | **paper** | **0.911** | **0.803** | — | **0.95** |
| sample_seq1 | improved | 1.000 | 1.000 | 9/10 | 0.95 |
| sample_seq2 | improved | 0.981 | 0.950 | 10/10 | 0.97 |
| sample_seq3 | improved | 0.941 | 0.928 | 24/27 | 0.93 |
| sample_seq4 | improved | 0.935 | 0.943 | 13/15 | 0.91 |
| sample_seq5 | improved | 0.973 | 0.938 | 9/8 | 0.98 |
| **mean** | **improved** | **0.966** | **0.952** | — | **0.95** |

The faithful `paper` method already reproduces the released trees well (mean ARI
≈ 0.91; perfect on seq1). Its main residual error is mild **over-segmentation**:
weighted-Jaccard@0.5 occasionally leaves a partial cross-view mask just below the
threshold, so an object splits into two instances (seq2 15 vs 10, seq5 16 vs 8 —
the downstream consumer's min-size filter removes most of these small fragments).
The optional `improved` overlap-coefficient term closes that gap (instance counts
and IoU much closer to the release; mean ARI ≈ 0.97), but it is a small recall
tweak on top of a faithful, already-correct reproduction — not a different method.
The **default is `paper`** so the baseline matches the publication; pass
`--variant improved` for the higher-recall variant.

For the update frames the per-frame containment **leaf sets** match the shipped
trees exactly (leaf Jaccard = 1.0 on seq1/seq2); only the count of internal
containment edges differs, controlled by `--containment-thresh`.

(Numbers above are from one run of the script on this fork; expect ±1–2 instances
of run-to-run variation because the DuoduoCLIP image embeddings are computed in
fp16. Re-run the script to regenerate them.)

## License

This repository is released under the Clutt3R-Seg Non-Commercial
Source-Available License. See `LICENSE` for details.

Third-party code, checkpoints, datasets, models, and generated assets are
governed by their own licenses. See `THIRD_PARTY.md` for third-party notices.

## Citation

If you found our work useful, please cite:

```bibtex
@inproceedings{noh2026clutt3rseg,
  title={Clutt3R-Seg: Sparse-view 3D Instance Segmentation for Language-grounded Grasping in Cluttered Scenes},
  author={Noh, Jeongho and Rhee, Tai Hyoung and Lee, Eunho and Kim, Jeongyun and Lee, Sunwoo and Kim, Ayoung},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```
