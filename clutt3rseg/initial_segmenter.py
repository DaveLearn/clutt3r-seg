"""Initial scene segmentation entry point for Clutt3R-Seg."""


from .utils import (
    load_depth, _bbox, backproject, refine_masks_by_depth, _letterbox,
    distinct_palette,
    mask_filter_dense_instances, filter_invalid_instances_by_superpoint_stats, reconstruct_instances_from_superpoints,
    export_prompt_pointcloud,
)
from .clip_backends.duoduo import DEFAULT_DUODUO_CHECKPOINT, load_duoduo_clip
from .tree_artifacts import (
    load_initial_ground_embedding,
    load_initial_leaf2inst,
    load_instance_tree_artifact,
)
import cv2
import numpy as np
import open3d as o3d
from pathlib import Path
import json
from PIL import Image
import glob
import pickle, argparse, torch

FRAME_OFFSET = 100
TOTAL_STAGES = 5


def _print_stage(step: int, message: str) -> None:
    print(f"[initial {step}/{TOTAL_STAGES}] {message}", flush=True)


def validate_initial_dataset(args) -> None:
    data_dir = Path(args.experiment_data_dir) / "data"
    transforms_path = data_dir / "transforms.json"
    if not transforms_path.exists():
        raise FileNotFoundError(
            f"Missing {transforms_path}. The experiment directory must contain a data/transforms.json file."
        )

    meta = json.load(open(transforms_path))
    frames_meta = meta.get("frames", [])
    if not frames_meta:
        raise ValueError(f"{transforms_path} does not contain any frames.")

    instance_mask_path = data_dir / "instance_masks"
    instance_tree_path = data_dir / "instance_tree.json"

    missing = []
    if not instance_mask_path.exists():
        missing.append(
            f"missing Grounded-SAM instance masks directory: {instance_mask_path}"
        )
    if not instance_tree_path.exists():
        missing.append(
            f"missing precomputed instance-tree artifact: {instance_tree_path}"
        )

    for fidx in args.initial_idx:
        if fidx < 0 or fidx >= len(frames_meta):
            missing.append(f"frame index {fidx} is outside transforms.json range 0..{len(frames_meta) - 1}")
            continue

        frame = frames_meta[fidx]
        rgb_path = data_dir / Path(frame["file_path"]).with_suffix(".png")
        depth_path = data_dir / Path(frame["depth_file_path"]).with_suffix(".png")
        mask_files = list(instance_mask_path.glob(f"mask_{fidx:06d}_*.png"))

        if not rgb_path.exists():
            missing.append(f"missing RGB image: {rgb_path}")
        if not depth_path.exists():
            missing.append(f"missing depth image: {depth_path}")
        if instance_mask_path.exists() and not mask_files:
            missing.append(f"missing instance masks for frame {fidx}: {instance_mask_path}/mask_{fidx:06d}_*.png")

    if missing:
        details = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Dataset validation failed. The active data/depth directory should contain MVSAnywhere depth, "
            "data/instance_masks should contain Grounded-SAM masks, and data/instance_tree.json should contain "
            "precomputed release tree assignments. For new sequences, generate a compatible tree artifact "
            f"with the full internal pipeline before using this source-available release.\n  - {details}"
        )


def initial_segmentation_consistency(args, clip, device = "cuda"):
    _print_stage(3, "Building initial geometry and superpoints")
    initial_scene_root = Path(args.experiment_data_dir) / "data"
    meta = json.load(open(initial_scene_root / "transforms.json"))
    frames_meta = meta["frames"]
    K = np.array([[meta["fl_x"], 0, meta["cx"]],
                  [0, meta["fl_y"], meta["cy"]],
                  [0, 0, 1]], np.float32)

    initial_frames_meta = [frames_meta[i] for i in args.initial_idx]


    xyz_full, rgb_full, mid_full = [], [], []
    from collections import defaultdict
    inst2bin_mask = {}
    inst2pids = defaultdict(list)
    inst2pcds = defaultdict(list)

    offset = 0

    raw_images = {}
    output_dir = args.experiment_data_dir / "output"
    instance_mask_path = args.experiment_data_dir / "data" / "instance_masks"

    for local_idx, fr in enumerate(initial_frames_meta):

        f_idx_global = args.initial_idx[local_idx]
        rgb_path   = initial_scene_root / Path(fr["file_path"]).with_suffix(".png")
        depth_path = initial_scene_root / Path(fr["depth_file_path"]).with_suffix(".png")
        if not rgb_path.exists():
            continue
        rgb   = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB) / 255.
        raw_images[f_idx_global] = Image.open(rgb_path).convert("RGB")
        H_assert, W_assert = rgb.shape[:2]
        depth = load_depth(depth_path, args.depth_scale)
        pts_cam, uvs = backproject(depth, K, args.max_depth)
        T = np.asarray(fr["transform_matrix"], np.float32)
        pts_wld = (T[:3,:3] @ pts_cam.T + T[:3,3:4]).T

        h, w = depth.shape

        idx_img = np.full((h, w), -1, np.int32)
        idx_img[uvs[:,1], uvs[:,0]] = np.arange(len(uvs))
        label_img = np.zeros((h, w), np.int32)
        mask_glob = glob.glob(str((instance_mask_path) / f"mask_{f_idx_global:06d}_*.png"))
        mask_refined = refine_masks_by_depth(mask_glob, depth, mode = "naive")


        for inst_id, mask in mask_refined.items():
            gid = inst_id + f_idx_global * FRAME_OFFSET
            lid = inst_id

            if mask.sum() > 10:
                y0, y1, x0, x1 = _bbox(mask)
                crop_rgb  = raw_images[f_idx_global].crop((x0, y0, x1 + 1, y1 + 1))
                crop_mask = (mask[y0:y1 + 1, x0:x1 + 1] * 255).astype(np.uint8)
                crop_arr  = _letterbox(crop_rgb, crop_mask)
                inst2bin_mask[(f_idx_global, lid)] = crop_arr
            label_img[mask] = gid
            ys, xs = np.where(mask)
            if ys.size:
                idx_pts = idx_img[ys, xs]
                valid   = idx_pts != -1
                if np.any(valid):
                    idx_pts_valid = idx_pts[valid]

                    idx_global = (offset + idx_pts_valid).tolist()
                    inst2pids[(f_idx_global, lid)].append(idx_global)


                    colors = rgb[ys[valid], xs[valid]]
                    pc = np.hstack([pts_wld[idx_pts_valid], colors])
                    inst2pcds[(f_idx_global, lid)].append(pc)

        xyz_full.append(pts_wld)
        rgb_full.append(rgb[uvs[:,1], uvs[:,0]])
        mid_full.append(label_img[uvs[:,1], uvs[:,0]])
        offset += len(pts_wld)
        if len(pts_wld) != H_assert * W_assert:
            # Measured/sensor depth has holes (invalid pixels), unlike the dense
            # MVSAnywhere depth the paper assumes. This is tolerated: backprojection
            # already keeps only valid pixels, idx_img maps the rest to -1, and the
            # per-mask code below filters those out (valid = idx_pts != -1) with a
            # consistent global offset. So we only warn instead of failing, to stay
            # comparable to the other baselines which run on measured depth.
            invalid = H_assert * W_assert - len(pts_wld)
            print(
                f"[warn] Frame {f_idx_global}: {invalid} invalid/holed depth pixels "
                f"({100.0 * invalid / (H_assert * W_assert):.1f}%); proceeding on valid pixels only.",
                flush=True,
            )

    xyz_full = np.concatenate(xyz_full, 0)
    rgb_full = np.concatenate(rgb_full, 0)
    mid_full = np.concatenate(mid_full, 0)


    for key, pc_list in list(inst2pcds.items()):

        inst2pcds[key] = np.concatenate(pc_list, axis=0)

    pcd_rgb = o3d.geometry.PointCloud()
    pcd_rgb.points = o3d.utility.Vector3dVector(xyz_full)
    pcd_rgb.colors = o3d.utility.Vector3dVector(rgb_full)
    voxel_sz  = args.voxel_size
    DOWNSAMPLE_METHOD = "custom"
    if DOWNSAMPLE_METHOD == "custom":
        min_points = getattr(args, "min_points", 3)

        pts_full = np.asarray(pcd_rgb.points)
        cols_full = np.asarray(pcd_rgb.colors)


        vox_inds = np.floor(pts_full / voxel_sz).astype(np.int32)


        import pandas as pd
        df = pd.DataFrame({
            'vx': vox_inds[:, 0], 'vy': vox_inds[:, 1], 'vz': vox_inds[:, 2],
            'px': pts_full[:, 0], 'py': pts_full[:, 1], 'pz': pts_full[:, 2],
            'cr': cols_full[:, 0], 'cg': cols_full[:, 1], 'cb': cols_full[:, 2],
        })


        grouped = df.groupby(['vx', 'vy', 'vz'])
        aggregated = grouped.agg({
            'px': 'sum', 'py': 'sum', 'pz': 'sum',
            'cr': 'sum', 'cg': 'sum', 'cb': 'sum',
            'vx': 'size'
        }).rename(columns={'vx': 'counts'})


        filtered = aggregated[aggregated['counts'] >= min_points].copy()

        dense = len(filtered)
        sparse = len(aggregated) - dense


        counts = filtered['counts'].values
        centroids = filtered[['px', 'py', 'pz']].values / counts[:, None]
        colours = filtered[['cr', 'cg', 'cb']].values / counts[:, None]

        down_pts = centroids.astype(np.float32)
        down_cols = colours.astype(np.float32)


        filtered['new_idx'] = np.arange(dense)


        df = df.join(filtered['new_idx'], on=['vx', 'vy', 'vz'])


        orig2down = df['new_idx'].fillna(-1).to_numpy(dtype=np.int32)

        pcd_down = o3d.geometry.PointCloud()

        pcd_down.points = o3d.utility.Vector3dVector(down_pts)
        pcd_down.colors = o3d.utility.Vector3dVector(down_cols)
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3*voxel_sz, max_nn=30))

        xyz_down     = np.asarray(pcd_down.points)
        rgb_down     = np.asarray(pcd_down.colors)
        normals_down = np.asarray(pcd_down.normals)


        pcd_down_rgb = o3d.geometry.PointCloud()
        pcd_down_rgb.points = o3d.utility.Vector3dVector(xyz_down)
        pcd_down_rgb.colors = o3d.utility.Vector3dVector(rgb_down)
        o3d.io.write_point_cloud(output_dir / "downsampled_rgb.ply", pcd_down_rgb, write_ascii=False)
    else:
        raise NotImplementedError("Down-sampling method not implemented")

    k = 8
    tree   = o3d.geometry.KDTreeFlann(pcd_down)
    parent = np.arange(len(xyz_down))
    size   = np.ones(len(xyz_down))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    _xyz = xyz_down
    _nrm = normals_down
    lam  = args.gc_lambda

    for i, pt in enumerate(_xyz):
        _, idx, _ = tree.search_knn_vector_3d(pt, k + 1)
        nbr = np.asarray(idx, dtype=np.int32)[1:]
        if nbr.size == 0:
            continue

        di = find(i)
        xi = pt
        ni = _nrm[i]

        for j in nbr:
            dj = find(j)
            if di == dj:
                continue
            v = _xyz[j] - xi
            dist = float(np.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]))
            nj = _nrm[j]
            dot = float(ni[0]*nj[0] + ni[1]*nj[1] + ni[2]*nj[2])
            w = 0.5 * dist + (1.0 - dot)
            if w < lam:
                if size[di] < size[dj]:
                    di, dj = dj, di
                parent[dj] = di
                size[di]  += size[dj]
                di = find(di)

    root2lbl = {}
    labels   = np.empty(len(_xyz), np.int32)
    cur_lbl  = 0
    for i in range(len(_xyz)):
        r = find(i)
        if r not in root2lbl:
            root2lbl[r] = cur_lbl
            cur_lbl += 1
        labels[i] = root2lbl[r]

    n_sp = labels.max() + 1
    down2sp = labels.copy()

    rng = np.random.default_rng(seed=12345)
    sp_random_col = rng.uniform(0, 1, size=(n_sp, 3))
    pcd_down.colors = o3d.utility.Vector3dVector(sp_random_col[down2sp])
    uniques, counts = np.unique(down2sp, return_counts=True)
    counts_per_sp = counts


    bins = [(1,1), (2,2), (3,5), (6,10), (11,25), (26,50), (51,100), (101,200), (201,300)]
    for low, high in bins:
        if low == high:
            num = np.sum(counts_per_sp == low)
        else:
            num = np.sum((counts_per_sp >= low) & (counts_per_sp <= high))
    num_large = np.sum(counts_per_sp > bins[-1][1])
    sp_weights = {sp: count for sp, count in enumerate(counts_per_sp)}
    ground_sp_id = max(sp_weights, key=sp_weights.get)

    inst_mask2sp_stats = {}

    for (fidx, lid), _ in inst2pcds.items():
        orig_idxs_list = inst2pids.get((fidx, lid), [])
        if not orig_idxs_list or not orig_idxs_list[0]:
            continue


        orig_idxs = np.array(orig_idxs_list[0])


        valid_down_idxs_all = orig2down[orig_idxs]
        valid_down_idxs_all = valid_down_idxs_all[valid_down_idxs_all != -1]

        if valid_down_idxs_all.size == 0:
            continue


        unique_down_idxs = np.unique(valid_down_idxs_all)


        sp_ids = down2sp[unique_down_idxs]


        unique_sps, counts = np.unique(sp_ids, return_counts=True)
        sp_counts = dict(zip(unique_sps, counts))


        inst_mask2sp_stats[(fidx, lid)] = sp_counts


        for sp_id, cnt in sp_counts.items():
            occ_ratio = cnt / counts_per_sp[sp_id] if counts_per_sp[sp_id] > 0 else 0
            if not (0 <= occ_ratio <= 1):

                if not (0 <= occ_ratio <= 1.00001):
                    raise ValueError(f"Occupancy ratio {occ_ratio:.4f} for sp_id {sp_id} is out of bounds [0, 1]")


    _print_stage(4, "Computing mask embeddings and loading instance-tree artifact")
    emb_feats = {}

    clip.eval()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    use_amp = True
    bs_clip = getattr(args, "clip_bs", 1)

    batch_keys = sorted(inst2bin_mask.keys(), key=lambda x: (x[0], x[1]))
    if batch_keys:
        crops_np = np.stack([inst2bin_mask[k] for k in batch_keys], axis=0)
        N = crops_np.shape[0]

        batch_embeddings_cpu = np.empty((N, ), dtype=object)
        with torch.inference_mode():
            for s in range(0, N, bs_clip):
                e = min(s + bs_clip, N)
                batch_np = crops_np[s:e]
                if use_amp:
                    with torch.cuda.amp.autocast():
                        emb_t = clip.encode_image(batch_np)
                else:
                    emb_t = clip.encode_image(batch_np)
                emb_t = emb_t.float()
                emb_t = torch.nn.functional.normalize(emb_t, dim=1)
                emb_b = emb_t.cpu().numpy()
                for i in range(emb_b.shape[0]):
                    batch_embeddings_cpu[s + i] = emb_b[i]

        for i, (fidx, lid) in enumerate(batch_keys):
            emb = batch_embeddings_cpu[i]
            emb_feats[(fidx, lid)] = {"embedding": emb}

    tree_artifact = load_instance_tree_artifact(args.experiment_data_dir)
    leaf2inst_graph = load_initial_leaf2inst(tree_artifact, args.initial_idx)
    leaf2inst_graph = {
        key: inst_id
        for key, inst_id in leaf2inst_graph.items()
        if key in inst_mask2sp_stats and key in emb_feats
    }
    if not leaf2inst_graph:
        raise RuntimeError(
            "No valid initial leaves from instance_tree.json matched the current masks. "
            "Check that the artifact belongs to this sequence and initial frame set."
        )

    final_instance_ids = set(leaf2inst_graph.values())
    num_inst = max(final_instance_ids) + 1

    sp2inst_count: dict[int, dict[int, int]] = {}

    for (fidx, lid), inst_id in leaf2inst_graph.items():
        sp_stats = inst_mask2sp_stats.get((fidx, lid), {})
        for sp_id, cnt in sp_stats.items():
            sp2inst_count.setdefault(sp_id, {})
            sp2inst_count[sp_id][inst_id] = sp2inst_count[sp_id].get(inst_id, 0) + cnt


    sp2inst: dict[int, int] = {}
    for sp_id, cnt_dict in sp2inst_count.items():

        inst_sel = max(cnt_dict.items(), key=lambda x: (x[1], -x[0]))[0]
        sp2inst[sp_id] = inst_sel


    sp2inst[ground_sp_id] = -1


    instance_pcds = {}


    for i, sp_id in enumerate(down2sp):
        inst_id = sp2inst.get(sp_id, -1)
        if inst_id != -1:
            if inst_id not in instance_pcds:
                instance_pcds[inst_id] = []


            point_xyz = xyz_down[i]
            point_rgb = rgb_down[i]
            point_data = np.concatenate([point_xyz, point_rgb])
            instance_pcds[inst_id].append(point_data)


    for inst_id in instance_pcds:
        if len(instance_pcds[inst_id]) > 0:
            instance_pcds[inst_id] = np.array(instance_pcds[inst_id])
        else:

            del instance_pcds[inst_id]

    inst_colors = distinct_palette(num_inst)
    inst_colors_u8 = (inst_colors * 255).astype(np.uint8)


    points_to_keep = []
    colors_to_keep = []


    original_points = np.asarray(pcd_down.points)


    for i, sp_id in enumerate(down2sp):
        inst_id = sp2inst.get(sp_id, -1)


        if inst_id != -1:

            points_to_keep.append(original_points[i])

            colors_to_keep.append(inst_colors[inst_id])


    pcd_instances_only = o3d.geometry.PointCloud()
    pcd_instances_only.points = o3d.utility.Vector3dVector(np.array(points_to_keep))
    pcd_instances_only.colors = o3d.utility.Vector3dVector(np.array(colors_to_keep))


    out_path = output_dir / "3d_inst_seg.ply"
    o3d.io.write_point_cloud(out_path, pcd_instances_only, write_ascii=False)

    _print_stage(5, "Exporting target point cloud and preparing state")
    export_inst2all_points, orig_inst_labels = reconstruct_instances_from_superpoints(
        xyz_full=xyz_full,
        rgb_full=rgb_full,
        orig2down=orig2down,
        down2sp=down2sp,
        sp2inst=sp2inst,
    )
    export_inst2all_points = mask_filter_dense_instances(
        export_inst2all_points,
        leaf2inst_graph=leaf2inst_graph,
        frames_meta=frames_meta,
        instance_mask_path=instance_mask_path,
        K=K,
        erode_px=3,
        min_view_count=1,
        min_view_ratio=0.5,
        verbose=False,
    )
    to_drop = filter_invalid_instances_by_superpoint_stats(
        sp2inst=sp2inst,
        counts_per_sp=counts_per_sp,
        min_down_points=80,
        min_unique_sp=3,
        min_orig_points=300,
        export_after_mask=export_inst2all_points,
        verbose=False,
    )


    for inst_id in list(export_inst2all_points.keys()):
        if inst_id in to_drop:
            del export_inst2all_points[inst_id]

    instance_embeddings: dict[int, np.ndarray] = {}
    inst2crops: dict[int, list[np.ndarray]] = defaultdict(list)


    for (fidx, lid), crop_arr in inst2bin_mask.items():
        inst_id = leaf2inst_graph.get((fidx, lid))
        if inst_id is None:
            continue
        inst2crops[inst_id].append(crop_arr)


    for inst_id, crop_list in inst2crops.items():

        mv_imgs = np.stack(crop_list, axis=0).astype(np.uint8)
        emb = clip.encode_image(mv_imgs).squeeze(0).cpu().numpy()
        emb /= np.linalg.norm(emb) + 1e-8
        instance_embeddings[inst_id] = emb

    mean_ground_embedding = load_initial_ground_embedding(tree_artifact)

    target_prompt_emb = clip.encode_text(args.target_prompt)
    target_prompt_emb = target_prompt_emb.squeeze(0).cpu().numpy()
    inst2target_sim = {
        inst_id: float(np.dot(emb, target_prompt_emb))
        for inst_id, emb in instance_embeddings.items()
    }
    if not inst2target_sim:
        raise RuntimeError("No instance embeddings were computed for prompt-target export.")

    target_id = max(inst2target_sim, key=inst2target_sim.get)
    if target_id not in export_inst2all_points:
        raise KeyError(f"Target instance {target_id} is missing from point-cloud export data.")

    target_points = export_inst2all_points[target_id]
    export_prompt_pointcloud(
        target_points[:, :3],
        target_points[:, 3:6],
        output_dir,
        args.target_prompt,
    )

    original_data = {
        "inst2all_points": export_inst2all_points,
    }
    initial_data = {
        "instance_embeddings": instance_embeddings,
        "mean_ground_embedding": mean_ground_embedding,
        "inst_colors_u8": inst_colors_u8,
        "node2inst": leaf2inst_graph,
        "instance_pcds": instance_pcds,
        "original_data": original_data,
    }
    return initial_data


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.experiment_data_dir:
        raise ValueError("experiment_data_dir is not provided")

    _print_stage(1, "Validating input dataset")
    validate_initial_dataset(args)

    _print_stage(2, "Loading DuoduoCLIP model")
    clip = load_duoduo_clip(
        checkpoint=args.clip_checkpoint,
        device=device,
        duoduo_root=args.duoduo_root,
    )


    initial_data = initial_segmentation_consistency(args, clip=clip, device=device)

    state_path = args.experiment_data_dir / "state.pkl"
    args.experiment_data_dir.mkdir(parents=True, exist_ok=True)
    with open(state_path, 'wb') as f:
        pickle.dump(initial_data, f)

    return

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--experiment_data_dir", type=Path, help="Directory to save state.pkl and other experiment data.")
    pa.add_argument("--target_prompt", type=str, required=True, help="Target prompt for segmentation, e.g. 'a red cup'")
    pa.add_argument("--voxel_size", type=float, default=0.01)
    pa.add_argument("--depth_scale", type=float, default=0.001)
    pa.add_argument("--max_depth", type=float, default=5.0)
    pa.add_argument('--initial_idx', type=int, required=True, nargs='+', help="Initial frame indices to process, 0~7")
    pa.add_argument("--gc_lambda", type=float, default=0.010,
                   help="Graph construction threshold, lower = more edges")
    pa.add_argument("--duoduo-root", "--duoduo_root", dest="duoduo_root", type=Path, default=None,
                   help="Path to an external DuoduoCLIP checkout. Defaults to DUODUOCLIP_ROOT.")
    pa.add_argument("--clip-checkpoint", "--clip_checkpoint", dest="clip_checkpoint", type=str,
                   default=DEFAULT_DUODUO_CHECKPOINT,
                   help="DuoduoCLIP checkpoint filename or path accepted by the upstream wrapper.")
    args = pa.parse_args()
    output_dir = args.experiment_data_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    main(args)
