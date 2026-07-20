"""Reconstruction of the Clutt3R-Seg instance-tree builder from the paper.

The public release ships the *consumer* of ``data/instance_tree.json`` but not the
builder that produces it ("parts of that implementation depend on closed or
restricted components"). This module reimplements the builder from the method
description in the paper (arXiv:2602.11660, Sec. III) so the artifact can be
regenerated for new sequences.

Pipeline (initial segmentation, producing ``initial.leaf2inst``):

1. **Substrate** (:mod:`clutt3rseg.scene_substrate`): back-project + 5 mm voxel
   downsample + normal/distance super-voxels; per-mask super-voxel occupancy.
2. **Per-frame instance tree**: organise each frame's Grounded-SAM masks by 2D
   containment (mask B is a child of mask A if B's pixels are contained in A).
   Only *leaf* masks can be proper segments ("one-proper-per-path"), so leaves
   are the candidates for cross-view grouping.
3. **Cross-view grouping** (Algorithm 1): iterative greedy union of leaf nodes,
   first by super-voxel weighted-Jaccard spatial similarity
   ``S_spatial >= tau_spat`` (0.5), then the remaining nodes by DuoduoCLIP
   embedding cosine ``S_semantic >= tau_sem`` (0.65).
4. **Residual-node parent substitution**: leaves still ungrouped ("residual",
   typically over-segmentation) are folded into their parent when *all* of that
   parent's descendant leaves are residual, then a final spatial pass absorbs the
   consolidated node. This resolves over-segmentation.

Each converged group becomes one 3D instance; ``leaf2inst`` records, per
(frame, mask) leaf, its instance id. ``updates[frame]`` for additional frames is
just that frame's 2D containment tree (the consumer groups updates at run time
using stored embeddings), which we serialise from the same forest routine.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import glob
import json
import logging
from pathlib import Path
import re

import cv2
import numpy as np

from .scene_substrate import Leaf, SceneSubstrate, build_scene_substrate
from .tree_artifacts import ARTIFACT_NAME, leaf_id

logger = logging.getLogger("clutt3rseg-builder")

# Grouping thresholds.
TAU_SEMANTIC = 0.65
# A mask whose pixels are >= this fraction inside another mask is that mask's child.
CONTAINMENT_THRESH = 0.85
# A leaf dominated this much by the ground super-voxel is dropped before grouping.
GROUND_FRACTION = 0.5


@dataclass(frozen=True)
class GroupingConfig:
    """Knobs for the two-stage cross-view grouping (paper Algorithm 1).

    The faithful algorithm is agglomerative average-linkage clustering on a
    complete graph of per-frame-tree leaf nodes, with edges only between leaves of
    *different* frames (``phi(u) != phi(v)``): stage 1 greedily contracts the
    max-spatial edge while ``S_spatial >= tau_spat``, then stage 2 contracts the
    max-semantic edge while ``S_semantic >= tau_sem``; on each merge the new node's
    edge to a neighbour is the mean similarity over all constituent pairs
    (GroupAndRewire, lines 32-35).

    The single ``paper`` preset (see :data:`PAPER`) is the literal method section:
    the spatial term is the super-voxel **weighted Jaccard** (intersection-over-
    union), ``tau_spat=0.5``, ``tau_sem=0.65``, average linkage. The bare
    ``GroupingConfig()`` defaults are those values.
    """

    tau_spat: float = 0.5
    tau_sem: float = TAU_SEMANTIC
    linkage: str = "average"  # mean similarity over constituent pairs (Lance-Williams)
    containment_thresh: float = CONTAINMENT_THRESH


PAPER = GroupingConfig()

# Backwards-compatible module-level default (the paper spatial acceptance).
TAU_SPATIAL = PAPER.tau_spat


# --------------------------------------------------------------------------- #
# Per-frame 2D containment forest
# --------------------------------------------------------------------------- #
def build_containment_forest(
    masks: dict[int, np.ndarray],
    contain_thresh: float = CONTAINMENT_THRESH,
) -> tuple[dict[int, int], dict[int, list[int]], list[int], dict[int, list[int]]]:
    """Organise one frame's masks into a containment forest.

    Returns ``(parent_of, children, leaves, descendant_leaves)`` where
    ``parent_of[child] = immediate_parent`` (the smallest strictly-larger mask
    that contains the child), ``leaves`` are masks with no children, and
    ``descendant_leaves[node]`` lists the leaf masks under each internal node.
    """
    areas = {i: int(np.count_nonzero(m)) for i, m in masks.items()}
    ids = sorted(i for i in masks if areas[i] > 0)

    parent_of: dict[int, int] = {}
    for i in ids:
        ai = areas[i]
        best, best_area = None, None
        for j in ids:
            if i == j or areas[j] <= ai:
                continue
            inter = int(np.count_nonzero(np.logical_and(masks[i], masks[j])))
            if inter / ai >= contain_thresh and (best_area is None or areas[j] < best_area):
                best, best_area = j, areas[j]
        if best is not None:
            parent_of[i] = best

    children: dict[int, list[int]] = defaultdict(list)
    for child, parent in parent_of.items():
        children[parent].append(child)

    leaves = [i for i in ids if i not in children]

    def _descend(node: int) -> list[int]:
        if node not in children:
            return [node]
        out: list[int] = []
        for c in children[node]:
            out.extend(_descend(c))
        return out

    descendant_leaves = {p: _descend(p) for p in children}
    return parent_of, dict(children), leaves, descendant_leaves


# --------------------------------------------------------------------------- #
# Grouping (paper Algorithm 1: agglomerative average-linkage on a cross-frame
# leaf graph, spatial stage then semantic stage)
# --------------------------------------------------------------------------- #
def _pair_spatial(occ_a: dict[int, float], occ_b: dict[int, float], counts_per_sp: np.ndarray) -> float:
    """Weighted super-voxel spatial similarity between two leaf masks, occupancy capped at 1.

    This is the paper's term: the weighted **Jaccard**, intersection-over-**union**.
    """
    inter = union = 0.0
    for k in set(occ_a) | set(occ_b):
        cap = counts_per_sp[k]
        oa = min(1.0, occ_a.get(k, 0.0) / cap)
        ob = min(1.0, occ_b.get(k, 0.0) / cap)
        inter += cap * min(oa, ob)
        union += cap * max(oa, ob)
    return inter / (union + 1e-8) if union > 0 else 0.0


def _agglomerate(
    leaves: list[Leaf],
    spat: dict[tuple[int, int], float],
    sem: dict[tuple[int, int], float],
    tau_spat: float,
    tau_sem: float,
    linkage: str,
) -> tuple[list[list[int]], dict[str, int]]:
    """Paper Algorithm 1: greedy edge-contraction on a complete cross-frame leaf graph.

    ``spat``/``sem`` hold the per-edge similarities for cross-frame leaf-index
    pairs ``(i, j)`` with ``i < j`` (same-frame pairs are simply absent, enforcing
    ``phi(u) != phi(v)``). Stage 1 contracts the max-spatial edge while it is
    ``>= tau_spat``; stage 2 then the max-semantic edge while ``>= tau_sem``. On
    each contraction the merged node's similarity to every neighbour is the mean
    over all constituent leaf pairs (``linkage="average"``, GroupAndRewire), which
    equals the Lance-Williams average update. A merge is forbidden when the two
    clusters already share a frame (a single object appears once per frame).
    """
    members: dict[int, list[int]] = {i: [i] for i in range(len(leaves))}
    frames: dict[int, set[int]] = {i: {leaves[i][0]} for i in range(len(leaves))}
    size: dict[int, int] = {i: 1 for i in range(len(leaves))}
    active = set(range(len(leaves)))
    spat = dict(spat)
    sem = dict(sem)
    next_id = len(leaves)

    def okey(a: int, b: int) -> tuple[int, int]:
        return (a, b) if a < b else (b, a)

    def best(simdict: dict[tuple[int, int], float], tau: float) -> tuple[int, int] | None:
        chosen, chosen_s = None, tau
        for (a, b), s in simdict.items():
            if s >= chosen_s and a in active and b in active:
                chosen, chosen_s = (a, b), s
        return chosen

    def contract(a: int, b: int) -> None:
        nonlocal next_id
        w = next_id
        next_id += 1
        members[w] = members[a] + members[b]
        frames[w] = frames[a] | frames[b]
        size[w] = size[a] + size[b]
        active.discard(a)
        active.discard(b)
        for c in list(active):
            if frames[w] & frames[c]:  # would put two leaves of one frame together
                continue
            for sd in (spat, sem):
                sa = sd.get(okey(a, c))
                sb = sd.get(okey(b, c))
                sa = 0.0 if sa is None else sa
                sb = 0.0 if sb is None else sb
                if linkage == "max":
                    sd[okey(w, c)] = max(sa, sb)
                else:  # average linkage (Lance-Williams over constituent leaves)
                    sd[okey(w, c)] = (size[a] * sa + size[b] * sb) / (size[a] + size[b])
        active.add(w)

    counts = {"spatial": 0, "semantic": 0}
    while True:
        pair = best(spat, tau_spat)
        if pair is None:
            break
        contract(*pair)
        counts["spatial"] += 1
    while True:
        pair = best(sem, tau_sem)
        if pair is None:
            break
        contract(*pair)
        counts["semantic"] += 1

    return [members[c] for c in active], counts


def _apply_residual_substitution(leaf2cluster: dict[Leaf, int], forests: dict[int, tuple]) -> int:
    """Fold over-segmented residual leaves into one cluster per parent (in place).

    A leaf is *residual* if it is still alone in its cluster after grouping. When
    every descendant leaf of an internal containment node is residual, those
    fragments are one over-segmented object, so we union them into one cluster.
    """
    sizes: dict[int, int] = {}
    for cid in leaf2cluster.values():
        sizes[cid] = sizes.get(cid, 0) + 1
    folds = 0
    for frame, (_po, _ch, _lv, descendant_leaves) in forests.items():
        for dleaves in descendant_leaves.values():
            present = [(frame, m) for m in dleaves if (frame, m) in leaf2cluster]
            if len(present) < 2 or not all(sizes[leaf2cluster[x]] == 1 for x in present):
                continue
            target = leaf2cluster[present[0]]
            for x in present[1:]:
                leaf2cluster[x] = target
                folds += 1
    return folds


def build_initial_leaf2inst(
    substrate: SceneSubstrate,
    embeddings: dict[Leaf, np.ndarray],
    config: GroupingConfig | None = None,
) -> tuple[dict[Leaf, int], dict[int, tuple], set[Leaf]]:
    config = config or PAPER
    counts_per_sp = substrate.counts_per_sp.astype(np.float64)
    ground = substrate.ground_sp_id

    forests = {
        f: build_containment_forest(masks, config.containment_thresh)
        for f, masks in substrate.frame_masks.items()
    }

    # Vertices of the leaf graph: per-frame-tree leaves, minus ground/empty masks.
    leaves: list[Leaf] = []
    occ_of: dict[Leaf, dict[int, float]] = {}
    ground_leaves: set[Leaf] = set()
    for frame, (_po, _ch, frame_leaves, _dl) in forests.items():
        for m in frame_leaves:
            leaf = (frame, m)
            counts = substrate.mask2sp_counts.get(leaf, {})
            total = sum(counts.values())
            gfrac = counts.get(ground, 0) / total if total else 0.0
            occ = {sp: float(c) for sp, c in counts.items() if sp != ground}
            if total and gfrac > GROUND_FRACTION:
                ground_leaves.add(leaf)
                continue
            if not occ and leaf not in embeddings:
                continue
            leaves.append(leaf)
            occ_of[leaf] = occ

    # Complete graph over cross-frame leaf pairs (phi(u) != phi(v)); each edge
    # stores spatial and semantic similarity.
    spat_edges: dict[tuple[int, int], float] = {}
    sem_edges: dict[tuple[int, int], float] = {}
    for i in range(len(leaves)):
        for j in range(i + 1, len(leaves)):
            if leaves[i][0] == leaves[j][0]:
                continue  # same frame -> no edge
            spat_edges[(i, j)] = _pair_spatial(occ_of[leaves[i]], occ_of[leaves[j]], counts_per_sp)
            ei, ej = embeddings.get(leaves[i]), embeddings.get(leaves[j])
            sem_edges[(i, j)] = float(np.dot(ei, ej)) if ei is not None and ej is not None else -1.0

    clusters, counts = _agglomerate(
        leaves, spat_edges, sem_edges, config.tau_spat, config.tau_sem, config.linkage
    )

    leaf2cluster = {leaves[idx]: cid for cid, idxs in enumerate(clusters) for idx in idxs}
    n_fold = _apply_residual_substitution(leaf2cluster, forests)

    # Re-number instances largest-first (by leaf count) for stable, readable ids.
    cluster_sizes: dict[int, int] = {}
    for cid in leaf2cluster.values():
        cluster_sizes[cid] = cluster_sizes.get(cid, 0) + 1
    order = {cid: rank for rank, cid in enumerate(sorted(cluster_sizes, key=lambda c: -cluster_sizes[c]))}
    leaf2inst = {leaf: order[cid] for leaf, cid in leaf2cluster.items()}

    logger.info(
        "Grouping[jaccard/%s]: %d leaves -> %d instances (spatial=%d, semantic=%d, residual-fold=%d)",
        config.linkage, len(leaves), len(set(leaf2inst.values())),
        counts["spatial"], counts["semantic"], n_fold,
    )
    return leaf2inst, forests, ground_leaves


# --------------------------------------------------------------------------- #
# DuoduoCLIP embeddings
# --------------------------------------------------------------------------- #
def compute_leaf_embeddings(crops: dict[Leaf, np.ndarray], clip) -> dict[Leaf, np.ndarray]:
    import torch

    out: dict[Leaf, np.ndarray] = {}
    clip.eval()
    with torch.inference_mode():
        for leaf in sorted(crops, key=lambda x: (x[0], x[1])):
            arr = crops[leaf][None, ...].astype(np.uint8)
            with torch.cuda.amp.autocast():
                emb = clip.encode_image(arr)
            emb = torch.nn.functional.normalize(emb.float(), dim=1)[0].cpu().numpy()
            out[leaf] = emb.astype(np.float32)
    return out


def _mean_ground_embedding(ground_leaves: set[Leaf], embeddings: dict[Leaf, np.ndarray]) -> list[float] | None:
    embs = [embeddings[leaf] for leaf in ground_leaves if leaf in embeddings]
    if not embs:
        return None
    mean = np.mean(np.stack(embs, 0), 0)
    n = np.linalg.norm(mean)
    return (mean / n).astype(np.float32).tolist() if n > 0 else None


# --------------------------------------------------------------------------- #
# Update-frame containment trees
# --------------------------------------------------------------------------- #
def _load_frame_masks(data_dir: Path, frame: int) -> dict[int, np.ndarray]:
    masks: dict[int, np.ndarray] = {}
    for mpath in glob.glob(str(data_dir / "instance_masks" / f"mask_{frame:06d}_*.png")):
        match = re.findall(r"mask_\d{6}_(\d+)\.png", mpath)
        if not match:
            continue
        m = cv2.imread(mpath, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            masks[int(match[0])] = m > 0
    return masks


def _serialize_update_tree(frame: int, parent_of: dict[int, int], leaves: list[int], descendant_leaves: dict[int, list[int]]) -> dict:
    return {
        "leaf_nodes": [{"frame": frame, "mask": int(m)} for m in sorted(leaves)],
        "parent_of": {leaf_id(frame, c): leaf_id(frame, p) for c, p in parent_of.items()},
        "descendant_leaves": {
            leaf_id(frame, p): [leaf_id(frame, m) for m in ms] for p, ms in descendant_leaves.items()
        },
    }


# --------------------------------------------------------------------------- #
# Top-level artifact assembly
# --------------------------------------------------------------------------- #
def build_instance_tree_artifact(
    experiment_data_dir: Path,
    initial_idx: list[int],
    clip=None,
    *,
    update_idx: list[int] | None = None,
    config: GroupingConfig | None = None,
    voxel_size: float = 0.005,
    depth_scale: float = 0.001,
    max_depth: float = 5.0,
    gc_lambda: float = 0.010,
    min_points: int = 3,
) -> dict:
    config = config or PAPER
    experiment_data_dir = Path(experiment_data_dir)
    substrate = build_scene_substrate(
        experiment_data_dir,
        initial_idx,
        voxel_size=voxel_size,
        depth_scale=depth_scale,
        max_depth=max_depth,
        gc_lambda=gc_lambda,
        min_points=min_points,
    )
    logger.info("Substrate: %d super-voxels, ground sp=%d", substrate.n_supervoxels, substrate.ground_sp_id)

    if clip is not None:
        embeddings = compute_leaf_embeddings(substrate.crops, clip)
    else:
        logger.warning("No DuoduoCLIP model provided; skipping the semantic grouping stage.")
        embeddings = {}

    leaf2inst, _forests, ground_leaves = build_initial_leaf2inst(substrate, embeddings, config)

    initial = {
        "initial_idx": [int(i) for i in initial_idx],
        "leaf2inst": [
            {"frame": int(f), "mask": int(m), "instance": int(inst)}
            for (f, m), inst in sorted(leaf2inst.items())
        ],
        "mean_ground_embedding": _mean_ground_embedding(ground_leaves, embeddings),
    }

    updates: dict[str, dict] = {}
    data_dir = experiment_data_dir / "data"
    for uf in update_idx or []:
        masks = _load_frame_masks(data_dir, uf)
        if not masks:
            logger.warning("Update frame %d has no instance masks; skipping.", uf)
            continue
        parent_of, _children, leaves, descendant_leaves = build_containment_forest(masks, config.containment_thresh)
        updates[str(int(uf))] = _serialize_update_tree(uf, parent_of, leaves, descendant_leaves)

    return {
        "schema_version": 1,
        "source": (
            "Instance-tree artifact reconstructed by clutt3rseg.tree_builder "
            f"(paper arXiv:2602.11660), grouping: spatial=jaccard "
            f"tau_spat={config.tau_spat} tau_sem={config.tau_sem} linkage={config.linkage}."
        ),
        "initial": initial,
        "updates": updates,
    }


def write_instance_tree_artifact(experiment_data_dir: Path, artifact: dict) -> Path:
    out_path = Path(experiment_data_dir) / "data" / ARTIFACT_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(artifact, f, indent=2)
    return out_path
